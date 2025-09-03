import os
import asyncio
from datetime import datetime, timedelta
from typing import List, Optional
from bson import ObjectId

from core.cqrs import CommandRegistry, CommandHandler, Mediator
from core.llm import QwenLLM
from core.result import Result
from shared.messages import ChatMessage, SettingMessage
from app.database import get_collections
from app.database.enums import ChatRoleType
from app.database.models import (
    ChatHistoryModel,
    ChatSessionModel,
    SettingModel,
    UserProfileModel,
    HealthRecordModel
)
from app.dto.models import ChatHistoryModelDTO
from utils import get_logger
from ..create_chat_command import CreateChatCommand
from shared.rag_templates import render_template

from app.feature.train_ai import GetRetrievedContextQuery


# Load environment
if not os.getenv("QWEN_MODEL"):
    from dotenv import load_dotenv
    load_dotenv()


@CommandRegistry.register_handler(CreateChatCommand)
class CreateChatCommandHandler(CommandHandler):
    def __init__(self):
        super().__init__()
        self.logger = get_logger(__name__)
        self.db = get_collections()
        self.llm_client = None
        self.TOTAL_TIMEOUT = 120
        self.LLM_TIMEOUT = 45

    async def get_llm_client(self) -> QwenLLM:
        if self.llm_client is None:
            self.llm_client = QwenLLM(
                model=os.getenv("QWEN_MODEL", "qwen2.5:3b-instruct"),
                base_url=os.getenv("QWEN_URL", "http://localhost:11434")
            )
        return self.llm_client

    async def _with_timeout(self, coro, timeout: float, op: str) -> any:
        try:
            start = asyncio.get_event_loop().time()
            result = await asyncio.wait_for(coro, timeout=timeout)
            elapsed = asyncio.get_event_loop().time() - start
            self.logger.debug(f"{op} hoàn thành trong {elapsed:.2f}s")
            return result
        except asyncio.TimeoutError:
            self.logger.error(f"TIMEOUT: {op} vượt quá {timeout}s")
            raise asyncio.TimeoutError(f"{op} timeout sau {timeout}s")
        except Exception as e:
            elapsed = asyncio.get_event_loop().time() - start
            self.logger.error(f"{op} thất bại sau {elapsed:.2f}s: {e}")
            raise

    async def create_session(
        self,
        user_id: str,
        title: str,
        session_id: Optional[str] = None
    ) -> Optional[ChatSessionModel]:
        try:
            if user_id == "admin":
                filter_ = {"user_id": "admin"}
                update = {"$setOnInsert": ChatSessionModel(user_id="admin", title="Test AI").to_dict()}
                await self.db.chat_sessions.update_one(filter_, update, upsert=True)
                doc = await self.db.chat_sessions.find_one(filter_)
                return ChatSessionModel.from_dict(doc)

            if session_id and ObjectId.is_valid(session_id):
                obj_id = ObjectId(session_id)
                doc = await self.db.chat_sessions.find_one({"_id": obj_id})
                if doc:
                    return ChatSessionModel.from_dict(doc)

            clean_title = (title or "Cuộc trò chuyện mới")[:100]
            session = ChatSessionModel(user_id=user_id, title=clean_title)
            result = await self.db.chat_sessions.insert_one(session.to_dict())
            session._id = result.inserted_id
            return session

        except Exception as e:
            self.logger.error(f"Lỗi tạo session: {e}", exc_info=True)
            return None

    async def update_session(self, session_id: str) -> bool:
        try:
            if not ObjectId.is_valid(session_id):
                return False
            obj_id = ObjectId(session_id)
            result = await self.db.chat_sessions.update_one(
                {"_id": obj_id},
                {"$set": {"updated_at": datetime.utcnow()}}
            )
            return result.modified_count > 0
        except Exception as e:
            self.logger.error(f"Cập nhật session thất bại: {e}", exc_info=True)
            return False

    async def get_histories(self, session_id: str) -> List[ChatHistoryModel]:
        try:
            if not ObjectId.is_valid(session_id):
                return []
            obj_id = ObjectId(session_id)
            cursor = self.db.chat_histories.find({"session_id": str(obj_id)}) \
                .sort("updated_at", 1).limit(20)
            docs = await cursor.to_list(length=20)
            histories = []
            for doc in docs:
                model = ChatHistoryModel.from_dict(doc)
                if isinstance(model.role, str):
                    model.role = ChatRoleType.USER if model.role.lower() == "user" else ChatRoleType.AI
                histories.append(model)
            return histories
        except Exception as e:
            self.logger.error(f"Không thể lấy lịch sử chat: {e}", exc_info=True)
            return []

    async def save_data(self, data: ChatHistoryModel) -> bool:
        try:
            data_dict = data.to_dict()
            if isinstance(data.session_id, ObjectId):
                data_dict["session_id"] = str(data.session_id)
            data_dict["updated_at"] = datetime.utcnow()
            result = await self.db.chat_histories.insert_one(data_dict)
            return result.acknowledged
        except Exception as e:
            self.logger.error(f"Lưu tin nhắn thất bại: {e}", exc_info=True)
            return False

    async def get_user_profile(self, user_id: str) -> Optional[UserProfileModel]:
        try:
            doc = await self.db.user_profiles.find_one({"user_id": user_id})
            return UserProfileModel.from_dict(doc) if doc else None
        except Exception as e:
            self.logger.error(f"Không thể lấy hồ sơ người dùng {user_id}: {e}")
            return None

    async def get_recent_health_records(
        self,
        user_id: str,
        record_type: str,
        top: int = 3
    ) -> List[HealthRecordModel]:
        try:
            seven_days_ago = datetime.utcnow() - timedelta(days=7)
            cursor = self.db.health_records.find({
                "user_id": str(user_id),
                "type": record_type,
                "timestamp": {"$gte": seven_days_ago}
            }).sort("timestamp", -1).limit(top)
            docs = await cursor.to_list(length=top)
            records = []
            for doc in docs:
                try:
                    model = HealthRecordModel.from_dict(doc)
                    records.append(model)
                except Exception as e:
                    self.logger.error(f"Lỗi tạo HealthRecordModel: {e}")
            return records
        except Exception as e:
            self.logger.error(f"Lỗi lấy chỉ số sức khỏe: {e}", exc_info=True)
            return []

    async def get_relevant_user_context(self, user_id: str, question: str) -> str:
        profile = await self.get_user_profile(user_id)
        if not profile or user_id == "admin":
            return ""

        q = question.lower()
        parts = [
            f"Bệnh nhân: {profile.full_name} (ID: {profile.patient_id}), {profile.age} tuổi, {profile.gender}, "
            f"tiểu đường loại {profile.diabetes_type}"
        ]

        if any(kw in q for kw in ["đường huyết", "glucose"]):
            records = await self.get_recent_health_records(user_id, "Đường huyết", top=3)
            if records:
                summary = ", ".join([f"{r.value:.1f} mmol/l" for r in records])
                parts.append(f"Đường huyết: {summary}")
            if profile.complications:
                parts.append(f"Biến chứng: {', '.join(profile.complications)}")

        if any(kw in q for kw in ["huyết áp", "blood pressure"]):
            records = await self.get_recent_health_records(user_id, "Huyết áp", top=2)
            if records:
                sys = [r for r in records if r.subtype == "Tâm thu"]
                if sys:
                    avg = sum(r.value for r in sys) / len(sys)
                    parts.append(f"Huyết áp: trung bình {avg:.0f} mmHg (Tâm thu)")

        if any(kw in q for kw in ["insulin", "tiêm"]):
            if profile.insulin_schedule:
                parts.append(f"Lịch tiêm insulin: {profile.insulin_schedule}")

        if any(kw in q for kw in ["ăn", "chế độ", "lối sống"]):
            if profile.lifestyle:
                parts.append(f"Lối sống: {profile.lifestyle}")
            if profile.bmi:
                parts.append(f"BMI: {profile.bmi:.1f}")

        return "\n".join(parts)

    async def classify_question_type(self, question: str, has_previous_trend: bool = False, ai_messages_count: int = 0) -> str:
        clean_question = question.strip()[:500]
        self.logger.info(f"BẮT ĐẦU PHÂN LOẠI CÂU HỎI: '{clean_question}'")
        q_lower = clean_question.lower()

        # Cụm từ nhận diện
        greeting_keywords = ["xin chào", "chào bạn", "chào bác", "hello", "hi ", "chào mừng"]
        question_keywords = ["muốn biết", "là gì", "gì không", "có gì", "bao gồm", "thế nào", "như thế nào", "gồm những gì"]
        trend_keywords = ["gần đây", "xu hướng", "thay đổi", "so sánh", "dạo này", "tuần trước"]
        personal_phrases = [
            "tôi bị", "của tôi", "tình trạng của tôi", "chỉ số của tôi",
            "tôi đang", "tôi cảm thấy", "tui muốn biết", "tôi muốn hỏi"
        ]
        health_keywords = ["đường huyết", "huyết áp", "tiểu đường", "insulin", "sức khỏe", "chỉ số"]

        # Nếu là lời chào đơn thuần
        is_greeting_only = (
            any(kw in q_lower for kw in greeting_keywords)
            and not any(kw in q_lower for kw in question_keywords)
        )
        if is_greeting_only:
            return "rag_only"

        # So sánh bệnh
        disease_keywords = ["tiểu đường", "ung thư", "tim mạch", "bệnh gan", "bệnh thận"]
        comparison_keywords = ["loại nào", "cái nào", "so sánh", "khác nhau", "nguy hiểm hơn", "tốt hơn"]
        if any(d in q_lower for d in disease_keywords) and any(c in q_lower for c in comparison_keywords):
            return "rag_only"

        # Trend: thời gian + chỉ số
        if any(kw in q_lower for kw in trend_keywords):
            if any(kw in q_lower for kw in health_keywords):
                return "trend"

        # Personal: chia sẻ cá nhân
        if any(phrase in q_lower for phrase in personal_phrases):
            if any(kw in q_lower for kw in health_keywords):
                return "personal"

        # Gọi LLM
        llm = await self.get_llm_client()
        prompt = f"""
Bạn là hệ thống phân loại câu hỏi y tế. Chỉ trả về: rag_only, personal, trend, invalid — không giải thích.

QUY TẮC PHÂN LOẠI
- invalid: nội dung tiêu cực, tự tử, bỏ điều trị
- trend: có yếu tố thời gian + chỉ số (vd: "gần đây", "xu hướng")
- personal: chia sẻ bản thân, chỉ số, triệu chứng (vd: "tôi bị", "của tôi")
- rag_only: câu hỏi kiến thức chung

LƯU Ý
- "Xin chào bạn" → rag_only
- "Tôi muốn biết tiểu đường là gì?" → rag_only
- "Chỉ số gần đây của tôi ổn không?" → personal
- "Đường huyết gần đây thế nào?" → trend
- Không phân loại là personal nếu chỉ dùng "tôi muốn biết" mà không có nội dung cá nhân
- Chỉ personal khi có chia sẻ cụ thể

Câu hỏi: "{clean_question}"
""".strip()

        llm_result = None
        try:
            response = await self._with_timeout(
                llm.generate(prompt=prompt, max_tokens=20, temperature=0.05),
                self.LLM_TIMEOUT,
                "Phân loại bằng LLM"
            )
            llm_result = response.strip().lower()
            self.logger.debug(f"LLM trả về: '{llm_result}'")
        except Exception as e:
            self.logger.warning(f"LLM classification failed: {e}")

        if llm_result == "personal":
            if any(phrase in q_lower for phrase in personal_phrases) and any(kw in q_lower for kw in health_keywords):
                return "personal"
            else:
                return "rag_only"

        if llm_result in ["rag_only", "trend", "invalid"]:
            return llm_result

        # Fallback
        if any(kw in q_lower for kw in ["chết", "tự tử", "bỏ thuốc", "không sống", "mệt quá"]):
            return "invalid"
        elif any(kw in q_lower for kw in trend_keywords):
            return "trend"
        elif any(phrase in q_lower for phrase in personal_phrases):
            return "personal"
        else:
            return "rag_only"

    async def _gen_rag_only_response(self, message: str, contexts: List[str], histories: List[ChatHistoryModel]) -> str:
        if not contexts:
            return await self._respond_with_empathy_and_guidance(message)

        try:
            with open("shared/rag_templates/system_prompt.txt", "r", encoding="utf-8") as f:
                system_prompt = f.read().strip()
        except Exception:
            system_prompt = "Bạn là chuyên gia y tế."

        cleaned_contexts = "\n\n".join([
            ctx.strip()
            .replace("[HEADING]", "### ").replace("[/HEADING]", "\n")
            .replace("[SUBHEADING]", "#### ").replace("[/SUBHEADING]", "\n")
            for ctx in contexts if ctx.strip()
        ])

        is_general = any(kw in message.lower() for kw in ["là gì", "gì là", "bao gồm những gì", "tổng quan"])
        if is_general:
            system_prompt += (
                "\n\nNếu câu hỏi là tổng quát, hãy bắt đầu bằng định nghĩa chung về bệnh tiểu đường, "
                "sau đó mới phân tích chi tiết."
            )

        try:
            prompt_text = render_template(
                template_name="rag_only.j2",
                system_prompt=system_prompt,
                contexts=cleaned_contexts,
                question=message,
                histories=histories
            )
        except Exception as e:
            self.logger.error(f"Template rendering failed: {e}")
            return "Xin lỗi, không thể tạo câu trả lời."

        llm = await self.get_llm_client()
        try:
            response = await self._with_timeout(
                llm.generate(prompt=prompt_text, max_tokens=1800),
                self.LLM_TIMEOUT,
                "Tạo phản hồi RAG"
            )
            return self._ensure_markdown(response.strip())
        except asyncio.TimeoutError:
            return "Xin lỗi, tôi đang bận. Vui lòng thử lại sau."
        except Exception:
            return "Xin lỗi, tôi đang xử lý."

    async def _gen_personalized_response(self, message: str, contexts: List[str], user_context: str, user_id: str, first_time: bool = True, histories: List[ChatHistoryModel] = None) -> str:
        profile = await self.get_user_profile(user_id)
        if not profile:
            return "Không tìm thấy hồ sơ người dùng."

        try:
            with open("shared/rag_templates/system_prompt.txt", "r", encoding="utf-8") as f:
                system_prompt = f.read().strip()
        except Exception:
            system_prompt = "Bạn là bác sĩ nội tiết."

        cleaned_contexts = "\n\n".join([ctx.strip() for ctx in contexts if ctx.strip()])

        try:
            prompt_text = render_template(
                template_name="personalized.j2",
                system_prompt=system_prompt,
                contexts=cleaned_contexts,
                user_context=user_context,
                question=message,
                full_name=profile.full_name,
                age=profile.age,
                first_time=first_time,
                histories=histories
            )
        except Exception as e:
            self.logger.error(f"Template rendering failed: {e}")
            return "Xin lỗi, không thể tạo câu trả lời."

        llm = await self.get_llm_client()
        try:
            response = await self._with_timeout(
                llm.generate(prompt=prompt_text, max_tokens=1800),
                self.LLM_TIMEOUT,
                "Tạo phản hồi cá nhân hóa"
            )
            return self._ensure_markdown(response.strip())
        except asyncio.TimeoutError:
            return "Xin lỗi, tôi đang xử lý. Vui lòng thử lại."
        except Exception:
            return "Xin lỗi, tôi đang xử lý."

    def _ensure_markdown(self, text: str) -> str:
        if not text or not text.strip():
            return "Tôi chưa thể tạo câu trả lời phù hợp."

        lower = text.lower()
        if any(kw in lower for kw in ["import ", "def ", "class ", "from ", "os.", "dotenv"]):
            return (
                "Hiện tôi chưa có tài liệu liên quan đến câu hỏi này.\n"
                "Hãy hỏi về đường huyết, insulin, chế độ ăn để tôi hỗ trợ tốt hơn."
            )

        import re
        text = re.sub(r'\*\*(.*?)\*\*', r'**\1**', text)
        text = re.sub(r'\*(.*?)\*', r'*\1*', text)

        lines = text.split('\n')
        cleaned = []
        in_thinking = False
        thinking_keywords = ["hãy suy nghĩ", "phân tích", "let me think", "step by step"]
        for line in lines:
            if any(kw in line.lower() for kw in thinking_keywords):
                in_thinking = True
                continue
            if line.startswith("### ") and in_thinking:
                in_thinking = False
            if not in_thinking and line.strip():
                cleaned.append(line)
        result = '\n'.join(cleaned).strip()

        if result.startswith(("I ", "You ", "Sorry", "cannot")):
            return "Xin lỗi, tôi chỉ hỗ trợ bằng tiếng Việt."
        return result or "Tôi chưa có thông tin để trả lời."

    async def generate_health_status_response(self, user_id: str, question: str, first_time: bool = True, has_previous_trend: bool = False) -> str:
        q = question.lower()
        time_keywords = ["gần đây", "xu hướng", "thay đổi", "dạo này", "tuần trước"]

        if not any(kw in q for kw in time_keywords):
            self.logger.warning(f"Câu hỏi không phải trend thực sự: '{question}'")
            return (
                "Câu hỏi của bạn là kiến thức chung, không liên quan đến xu hướng cá nhân.\n"
                "Tôi sẽ trả lời dưới dạng thông tin tổng quát."
            )

        profile = await self.get_user_profile(user_id)
        if not profile:
            return (
                "Tôi hiểu rằng việc chia sẻ thông tin cá nhân có thể khiến bạn cảm thấy không thoải mái, "
                "nhưng để tôi hỗ trợ bạn tốt nhất, bạn vui lòng cập nhật một số thông tin cơ bản như tuổi và loại bệnh lý.\n"
                "Chỉ cần vài phút thời gian — sẽ giúp tôi đưa ra lời khuyên chính xác và phù hợp với hoàn cảnh của bạn.\n"
                "Bạn không đơn độc trong hành trình này — tôi luôn ở đây để đồng hành."
            )

        glucose_records = await self.get_recent_health_records(user_id, "Đường huyết", top=5)
        bp_records = await self.get_recent_health_records(user_id, "Huyết áp", top=5)

        if not glucose_records and not bp_records:
            return (
                "Hiện tôi chưa thấy có dữ liệu sức khỏe gần đây.\n"
                "Hãy bắt đầu ghi lại đường huyết 1–2 lần mỗi ngày."
            )

        health_summary = ""
        full_name = profile.full_name
        age = profile.age

        if "huyết áp" in q:
            systolic_records = [r for r in bp_records if r.subtype == "Tâm thu"]
            if not systolic_records:
                return "Chưa có dữ liệu huyết áp Tâm thu để đánh giá."

            avg_sys = sum(r.value for r in systolic_records) / len(systolic_records)
            diastolic_records = [r for r in bp_records if r.subtype == "Tâm trương"]
            avg_dia = sum(r.value for r in diastolic_records) / len(diastolic_records) if diastolic_records else 0

            trend_sys = "tăng" if len(systolic_records) >= 2 and systolic_records[0].value > systolic_records[-1].value \
                else "giảm" if len(systolic_records) >= 2 and systolic_records[0].value < systolic_records[-1].value else "ổn định"

            if avg_sys > 140 or avg_dia > 90:
                bp_status = "cao – nguy cơ tim mạch tăng"
            elif avg_sys > 120 or avg_dia > 80:
                bp_status = "biên độ cao – cần theo dõi"
            else:
                bp_status = "ổn định"

            health_summary = (
                f"Huyết áp trung bình: {avg_sys:.0f}/{avg_dia:.0f} mmHg → mức {bp_status}\n"
                f"Đo gần nhất: {systolic_records[0].value:.0f}/{avg_dia:.0f} mmHg\n"
                f"Xu hướng: {trend_sys}"
            )
        elif any(kw in q for kw in ["đường huyết", "glucose"]):
            if not glucose_records:
                return "Chưa có dữ liệu đường huyết để đánh giá."

            values = [r.value for r in glucose_records]
            avg = sum(values) / len(values)
            latest = glucose_records[0].value
            trend = "tăng" if len(values) >= 2 and values[0] > values[-1] \
                else "giảm" if len(values) >= 2 and values[0] < values[-1] else "ổn định"

            if avg > 8.0:
                status = "cao – cần điều chỉnh"
            elif avg > 6.0:
                status = "trung bình – cần theo dõi"
            else:
                status = "thấp – cần cảnh báo hạ đường huyết"

            health_summary = (
                f"Đường huyết trung bình: {avg:.1f} mmol/l → mức {status}\n"
                f"Đo gần nhất: {latest:.1f} mmol/l\n"
                f"Xu hướng: {trend}"
            )
        else:
            parts = []
            if glucose_records:
                values = [r.value for r in glucose_records]
                avg = sum(values) / len(values)
                status = "cao" if avg > 8.0 else "trung bình" if avg > 6.0 else "thấp"
                trend = "tăng" if len(values) >= 2 and values[0] > values[-1] else "giảm" if len(values) >= 2 and values[0] < values[-1] else "ổn định"
                parts.append(f"Đường huyết: trung bình {avg:.1f} mmol/l → mức {status}, xu hướng {trend}")

            if bp_records:
                sys = [r for r in bp_records if r.subtype == "Tâm thu"]
                if sys:
                    avg_sys = sum(r.value for r in sys) / len(sys)
                    bp_status = "cao" if avg_sys > 140 else "biên độ cao" if avg_sys > 120 else "ổn định"
                    parts.append(f"Huyết áp: trung bình {avg_sys:.0f} mmHg → mức {bp_status}")

            health_summary = "\n".join(f"- {p}" for p in parts) if parts else "Không có dữ liệu để phân tích."

        user_context = await self.get_relevant_user_context(user_id, question)
        history_context = ""
        if not first_time:
            history_context = "Lần trước, tôi đã phân tích xu hướng cho bạn." if has_previous_trend \
                else "Bạn đã hỏi trước đó, nhưng chưa phân tích xu hướng chi tiết."

        try:
            prompt_text = render_template(
                template_name="trend_response.j2",
                user_context=user_context,
                health_summary=health_summary,
                question=question,
                full_name=full_name,
                age=age,
                first_time=first_time,
                has_previous_trend=has_previous_trend,
                history_context=history_context
            )
        except Exception as e:
            self.logger.error(f"Template trend_response.j2 failed: {e}")
            return f"Phân tích sức khỏe:\n{health_summary}"

        llm = await self.get_llm_client()
        try:
            response = await self._with_timeout(
                llm.generate(prompt=prompt_text, max_tokens=600),
                self.LLM_TIMEOUT,
                "Phân tích sức khỏe"
            )
            return self._ensure_markdown(response.strip())
        except asyncio.TimeoutError:
            return f"Phân tích nhanh:\n{health_summary}\nTôi đang xử lý. Vui lòng thử lại sau."
        except Exception as e:
            self.logger.error(f"LLM failed in health analysis: {e}")
            return f"Dữ liệu thô:\n{health_summary}"

    async def get_polite_response_for_invalid_question(self, question: str) -> str:
        return (
            "Tôi hiểu bạn có thể đang cảm thấy mệt mỏi, nhưng sức khỏe của bạn rất quan trọng.\n"
            "Hãy tìm sự hỗ trợ từ bác sĩ hoặc người thân – bạn không đơn độc."
        )

    async def _respond_to_greeting(self) -> str:
        return (
            "Chào bạn.\n\n"
            "Tôi là trợ lý AI hỗ trợ người bệnh tiểu đường.\n\n"
            "Bạn có thể hỏi tôi về:\n"
            "- Đường huyết, Huyết áp, Insulin\n"
            "- Chế độ ăn, lối sống lành mạnh\n"
            "- Theo dõi biến chứng\n"
            "- Cách kiểm soát tiểu đường\n\n"
            "Ví dụ: Hôm nay đường huyết của tôi cao, nên ăn gì?\n\n"
            "Tôi luôn ở đây để đồng hành cùng bạn."
        )

    async def _respond_with_empathy_and_guidance(self, question: str) -> str:
        q = question.lower()
        disease_map = {"ung thư": "ung thư", "tim mạch": "bệnh tim", "gan": "bệnh gan", "thận": "bệnh thận"}
        found_disease = next((v for k, v in disease_map.items() if k in q), None)

        if found_disease:
            return (
                f"Tôi hiểu bạn đang quan tâm đến {found_disease}.\n"
                "Hiện tôi chưa có tài liệu chuyên sâu về chủ đề này, vì tôi đang tập trung hỗ trợ người bệnh tiểu đường.\n"
                "Tuy nhiên, nếu bạn hoặc người thân đang đối mặt với tiểu đường, tôi có thể giúp:\n"
                "- Theo dõi đường huyết, huyết áp\n"
                "- Gợi ý chế độ ăn phù hợp\n"
                "- Phân tích xu hướng sức khỏe\n"
                "- Nhắc lịch uống thuốc, đo chỉ số\n\n"
                "Bạn có muốn tìm hiểu thêm về cách sống chung với tiểu đường không?"
            )

        if any(kw in q for kw in ["biến chứng", "nguy hiểm", "hậu quả"]):
            return (
                "Bệnh tiểu đường nếu không kiểm soát tốt có thể gây ra nhiều biến chứng nghiêm trọng.\n"
                "Tôi có thể giải thích cho bạn về:\n"
                "- Biến chứng mắt: bệnh võng mạc\n"
                "- Biến chứng thận: suy thận mạn\n"
                "- Biến chứng thần kinh: tê bì, loét chân\n"
                "- Biến chứng tim mạch: nhồi máu, tai biến\n\n"
                "Bạn muốn tôi giải thích về biến chứng nào trước?"
            )

        return (
            "Cảm ơn bạn đã tin tưởng chia sẻ.\n\n"
            "Tôi có thể giúp bạn với các chủ đề:\n"
            "- Đường huyết, Huyết áp, Insulin\n"
            "- Chế độ ăn cho người tiểu đường\n"
            "- Theo dõi và phòng ngừa biến chứng\n"
            "- Cách kiểm soát tiểu đường hiệu quả\n\n"
            "Bạn muốn bắt đầu từ điều gì?"
        )

    async def execute(self, command: CreateChatCommand) -> Result[None]:
        try:
            return await self._with_timeout(
                self._execute_internal(command),
                self.TOTAL_TIMEOUT,
                "Xử lý chat hoàn chỉnh"
            )
        except asyncio.TimeoutError:
            self.logger.error("TOÀN BỘ YÊU CẦU TIMEOUT")
            return Result.failure(
                code=ChatMessage.CHAT_CREATED_FAILED.code,
                message="Xin lỗi, xử lý quá lâu. Vui lòng thử lại."
            )
        except Exception as e:
            self.logger.error(f"Lỗi nghiêm trọng trong execute: {e}", exc_info=True)
            return Result.failure(
                code=ChatMessage.CHAT_CREATED_FAILED.code,
                message=ChatMessage.CHAT_CREATED_FAILED.message
            )

    async def _execute_internal(self, command: CreateChatCommand) -> Result[None]:
        try:
            if not command.user_id or not command.content or len(command.content.strip()) < 2:
                return Result.failure(message="user_id và content là bắt buộc")

            clean_content = command.content.strip()[:1000]
            q_lower = clean_content.lower()

            session = await self.create_session(command.user_id, clean_content, command.session_id)
            if not session:
                return Result.failure(message="Không tạo được session.")

            user_chat = ChatHistoryModel(
                session_id=str(session.id),
                user_id=command.user_id,
                content=clean_content,
                role=ChatRoleType.USER
            )
            await self.save_data(user_chat)

            histories = await self.get_histories(session.id)
            ai_messages = [msg for msg in histories if msg.role == ChatRoleType.AI]
            first_time = len(ai_messages) == 0
            has_previous_trend = any(
                kw in msg.content.lower()
                for msg in ai_messages
                for kw in ["xu hướng", "gần đây", "phân tích", "đánh giá"]
            )

            # Xử lý lời chào đơn thuần
            greeting_keywords = ["xin chào", "chào bạn", "chào bác", "hello", "hi ", "chào mừng", "chào buổi"]
            question_keywords = ["muốn biết", "là gì", "gì không", "có gì", "bao gồm", "thế nào", "như thế nào", "gồm những gì"]

            is_greeting_only = (
                any(kw in q_lower for kw in greeting_keywords)
                and not any(kw in q_lower for kw in question_keywords)
            )

            if is_greeting_only:
                gen_text = await self._respond_to_greeting()
            else:
                retrieval_query = GetRetrievedContextQuery(query=clean_content)
                retrieval_result: Result = await Mediator.send(retrieval_query)

                contexts = []
                if retrieval_result.success and retrieval_result.data is not None:
                    contexts = [dto.content for dto in retrieval_result.data]
                else:
                    self.logger.info(f"Không có tài liệu liên quan: {clean_content}")

                question_type = await self.classify_question_type(
                    clean_content,
                    has_previous_trend=has_previous_trend,
                    ai_messages_count=len(ai_messages)
                )
                self.logger.info(f"CÂU HỎI ĐƯỢC PHÂN LOẠI: '{question_type}'")

                gen_text = ""

                if question_type == "invalid":
                    gen_text = await self.get_polite_response_for_invalid_question(clean_content)
                elif question_type == "trend":
                    gen_text = await self.generate_health_status_response(command.user_id, clean_content, first_time, has_previous_trend)
                elif question_type == "personal":
                    user_context = await self.get_relevant_user_context(command.user_id, clean_content)
                    if user_context:
                        gen_text = await self.generate_health_status_response(command.user_id, clean_content, first_time, has_previous_trend)
                    elif contexts:
                        gen_text = await self._gen_rag_only_response(clean_content, contexts, histories)
                    else:
                        gen_text = await self._respond_with_empathy_and_guidance(clean_content)
                else:
                    if contexts:
                        gen_text = await self._gen_rag_only_response(clean_content, contexts, histories)
                    else:
                        gen_text = await self._respond_with_empathy_and_guidance(clean_content)

            gen_text = self._ensure_markdown(gen_text)

            ai_chat = ChatHistoryModel(
                session_id=str(session.id),
                user_id=command.user_id,
                content=gen_text,
                role=ChatRoleType.AI
            )
            await self.save_data(ai_chat)
            await self.update_session(session.id)

            dto = ChatHistoryModelDTO.from_model(ai_chat)
            return Result.success(
                code=ChatMessage.CHAT_CREATED.code,
                message=ChatMessage.CHAT_CREATED.message,
                data=dto
            )

        except Exception as e:
            self.logger.error(f"Lỗi trong _execute_internal: {e}", exc_info=True)
            return Result.failure(
                code=ChatMessage.CHAT_CREATED_FAILED.code,
                message=ChatMessage.CHAT_CREATED_FAILED.message
            )