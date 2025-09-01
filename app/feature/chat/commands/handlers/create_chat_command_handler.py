import os
import dotenv
import asyncio
from datetime import datetime
from typing import List, Optional
from bson import ObjectId

from core.cqrs import CommandRegistry, CommandHandler
from core.embedding import EmbeddingModel
from core.llm import QwenLLM
from core.result import Result
from rag.vector_store import VectorStoreManager
from rag.retrieval.retriever import Retriever
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

dotenv.load_dotenv()

@CommandRegistry.register_handler(CreateChatCommand)
class CreateChatCommandHandler(CommandHandler):
    def __init__(self):
        super().__init__()
        self.logger = get_logger(__name__)
        self.db = get_collections()
        self.vector_store_manager = VectorStoreManager()
        self.embedding_model = None
        self.llm_client = None
        self.retriever_cache = {}
        
        # Timeout settings
        self.RAG_TIMEOUT = 30  # RAG retrieval timeout
        self.LLM_TIMEOUT = 45  # LLM generation timeout  
        self.EMBEDDING_TIMEOUT = 20  # Embedding timeout
        self.DB_TIMEOUT = 10  # Database query timeout
        self.TOTAL_TIMEOUT = 120  # Total process timeout

    async def get_llm_client(self) -> QwenLLM:
        if self.llm_client is None:
            self.llm_client = QwenLLM(
                model=os.getenv("QWEN_MODEL", "qwen2.5:3b-instruct"),
                base_url=os.getenv("QWEN_URL", "http://localhost:11434")
            )
        return self.llm_client

    async def get_embedding_model(self) -> EmbeddingModel:
        if self.embedding_model is None:
            self.embedding_model = EmbeddingModel()
        return self.embedding_model
    
    async def _with_timeout(self, coro, timeout_seconds: int, operation_name: str):
        """Wrapper để thêm timeout cho các async operations"""
        try:
            start_time = asyncio.get_event_loop().time()
            result = await asyncio.wait_for(coro, timeout=timeout_seconds)
            elapsed = asyncio.get_event_loop().time() - start_time
            self.logger.debug(f"{operation_name} completed in {elapsed:.2f}s")
            return result
        except asyncio.TimeoutError:
            self.logger.error(f"TIMEOUT: {operation_name} exceeded {timeout_seconds}s")
            raise asyncio.TimeoutError(f"{operation_name} timeout after {timeout_seconds}s")
        except Exception as e:
            elapsed = asyncio.get_event_loop().time() - start_time
            self.logger.error(f"{operation_name} failed after {elapsed:.2f}s: {e}")
            raise

    async def create_session(
        self,
        user_id: str,
        title: str,
        session_id: str = None
    ) -> Optional[ChatSessionModel]:
        try:
            if user_id == "admin":
                doc = await self.db.chat_sessions.find_one({"user_id": "admin"})
                if doc:
                    return ChatSessionModel.from_dict(doc)
                session = ChatSessionModel(user_id="admin", title="Test AI")
                result = await self.db.chat_sessions.insert_one(session.to_dict())
                session._id = result.inserted_id
                return session

            if session_id:
                obj_id = ObjectId(session_id)
                doc = await self.db.chat_sessions.find_one({"_id": obj_id})
                if doc:
                    return ChatSessionModel.from_dict(doc)

            session_title = title[:100] + "..." if len(title) > 100 else title
            session = ChatSessionModel(user_id=user_id, title=session_title)
            result = await self.db.chat_sessions.insert_one(session.to_dict())
            session._id = result.inserted_id
            return session

        except Exception as e:
            self.logger.error(f"Error creating session: {e}", exc_info=True)
            return None

    async def update_session(self, session_id: str) -> bool:
        try:
            obj_id = ObjectId(session_id)
            result = await self.db.chat_sessions.update_one(
                {"_id": obj_id},
                {"$set": {"updated_at": datetime.utcnow()}}
            )
            return result.modified_count > 0
        except Exception as e:
            self.logger.error(f"Update session failed: {e}", exc_info=True)
            return False

    async def get_histories(self, session_id: str) -> List[ChatHistoryModel]:
        try:
            obj_id = ObjectId(session_id)
            cursor = self.db.chat_histories.find({"session_id": str(obj_id)}) \
                .sort("updated_at", -1).limit(20)
            docs = await cursor.to_list(length=20)
            histories = []
            for doc in docs:
                model = ChatHistoryModel.from_dict(doc)
                if isinstance(model.role, str):
                    model.role = ChatRoleType.USER if model.role.lower() == "user" else ChatRoleType.AI
                histories.append(model)
            return histories
        except Exception as e:
            self.logger.error(f"Cannot get chat history: {e}", exc_info=True)
            return []

    async def save_data(self, data: ChatHistoryModel) -> bool:
        try:
            if isinstance(data.session_id, ObjectId):
                data.session_id = str(data.session_id)
            result = await self.db.chat_histories.insert_one(data.to_dict())
            return result.acknowledged
        except Exception as e:
            self.logger.error(f"Save chat history failed: {e}", exc_info=True)
            return False

    def get_retriever(self, collections: List[str]) -> Retriever:
        key = ",".join(sorted(collections))
        if key not in self.retriever_cache:
            self.retriever_cache[key] = Retriever(
                collections=collections,
                vector_store_manager=self.vector_store_manager
            )
        return self.retriever_cache[key]

    async def rewrite_query_if_needed(self, query: str, histories: List[ChatHistoryModel]) -> str:
        if len(histories) < 2:
            return query
        pronouns = ["nó", "vậy", "đó", "kia"]
        if not any(p in query.lower() for p in pronouns):
            return query
        for msg in reversed(histories):
            if msg.role == ChatRoleType.AI:
                content = msg.content.lower()
                if "đái tháo đường" in content:
                    return query.replace("nó", "đái tháo đường").replace("đó", "đái tháo đường")
                elif "insulin" in content:
                    return query.replace("nó", "insulin")
        return query

    async def get_user_profile(self, user_id: str) -> Optional[UserProfileModel]:
        try:
            doc = await self.db.user_profiles.find_one({"user_id": user_id})
            return UserProfileModel.from_dict(doc) if doc else None
        except Exception as e:
            self.logger.error(f"Không thể lấy hồ sơ người dùng {user_id}: {e}")
            return None

    async def get_recent_health_records(self, user_id: str, record_type: str, top: int = 3) -> List[HealthRecordModel]:
        try:
            cursor = self.db.health_records.find({
                "user_id": user_id,
                "type": record_type
            }).sort("timestamp", -1).limit(top)
            docs = await cursor.to_list(length=top)
            return [HealthRecordModel.from_dict(doc) for doc in docs if doc]
        except Exception as e:
            self.logger.error(f"Lỗi lấy chỉ số: {e}")
            return []

    async def classify_question_type(self, question: str) -> str:
        llm = await self.get_llm_client()
        prompt = f"""
Bạn là hệ thống phân loại câu hỏi y tế.

Phân loại câu hỏi sau vào đúng loại:
- rag_only: Câu hỏi kiến thức chung, không liên quan đến cá nhân
- personal: Có từ như 'tôi', 'của tôi', 'tình trạng của tôi'
- trend: Có từ như 'gần đây', 'xu hướng', 'thống kê', 'trong 3 tháng'
- invalid: Câu hỏi không phù hợp, nguy hiểm

Chỉ trả về 1 từ: rag_only, personal, trend, hoặc invalid.

Câu hỏi: "{question}"
""".strip()

        try:
            response = await self._with_timeout(
                llm.generate(prompt=prompt, max_tokens=20, temperature=0.1),
                self.LLM_TIMEOUT,
                "Question Classification"
            )
            response = response.strip().lower()
            if response in ["rag_only", "personal", "trend", "invalid"]:
                return response
            return "rag_only"
        except Exception as e:
            self.logger.error(f"Classification failed: {e}")
            return "rag_only"

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
            records = await self.get_recent_health_records(user_id, "BloodGlucose", top=3)
            if records:
                summary = ", ".join([f"{r.value:.1f} mmol/l" for r in records])
                parts.append(f"Đường huyết: {summary}")
            if profile.complications:
                parts.append(f"Biến chứng: {', '.join(profile.complications)}")
        if any(kw in q for kw in ["huyết áp", "blood pressure"]):
            records = await self.get_recent_health_records(user_id, "BloodPressure", top=2)
            if records:
                sys = [r.value for r in records if r.subtype == "tâm thu"]
                if sys:
                    parts.append(f"Huyết áp: trung bình {sum(sys)/len(sys):.0f} mmHg")
        if any(kw in q for kw in ["insulin", "tiêm"]):
            if profile.insulin_schedule:
                parts.append(f"Lịch tiêm insulin: {profile.insulin_schedule}")
        if any(kw in q for kw in ["ăn", "chế độ", "lối sống"]):
            if profile.lifestyle:
                parts.append(f"Lối sống: {profile.lifestyle}")
            if profile.bmi:
                parts.append(f"BMI: {profile.bmi:.1f}")
        return "\n".join(parts)

    async def _retrieve_rag_context(self, query: str, histories: List[ChatHistoryModel], settings: SettingModel) -> List[str]:
        if not settings.list_knowledge_ids:
            return []

        try:
            # Rewrite query
            rewritten = await self.rewrite_query_if_needed(query, histories)
            
            # Get embedding with timeout
            embedding = await self.get_embedding_model()
            query_vector = await self._with_timeout(
                embedding.embed(rewritten),
                self.EMBEDDING_TIMEOUT,
                "Query Embedding"
            )
            
            # Vector search with timeout
            retriever = self.get_retriever(settings.list_knowledge_ids)
            results = await self._with_timeout(
                retriever.retrieve(query_vector, top_k=settings.top_k * 2),
                self.RAG_TIMEOUT,
                "Vector Search"
            )

            score_threshold = getattr(settings, "search_accuracy", 0.75)
            filtered = [hit for hit in results if hit["score"] >= score_threshold]

            if not filtered:
                self.logger.info(f"Không tìm thấy tài liệu đủ liên quan cho: '{rewritten}'")
                return []

            # Đưa tất cả kết quả cho LLM, không làm relevance check ở đây
            # LLM sẽ tự quyết định sử dụng thông tin nào phù hợp
            contexts = [
                hit["payload"]["content"]
                for hit in filtered
                if hit["payload"] and hit["payload"].get("content")
            ][:settings.top_k]
            
            self.logger.debug(f"✅ Found {len(contexts)} contexts for query: '{rewritten}'")
            return contexts

        except asyncio.TimeoutError as e:
            self.logger.error(f"RAG retrieval timeout: {e}")
            return []
        except Exception as e:
            self.logger.error(f"RAG retrieval failed: {e}")
            return []

    async def _gen_rag_only_response(self, message: str, contexts: List[str]) -> str:
        if not contexts:
            return (
                "**Hiện tôi chưa có tài liệu** liên quan đến câu hỏi này.\n\n"
                "Nếu bạn có câu hỏi về **đường huyết, insulin, chế độ ăn cho người tiểu đường**, "
                "tôi rất sẵn lòng hỗ trợ.\n\n"
                "Bạn cũng có thể cung cấp thêm chi tiết để tôi tìm hiểu kỹ hơn."
            )

        try:
            with open("shared/rag_templates/system_prompt.txt", "r", encoding="utf-8") as f:
                system_prompt = f.read().strip()
        except Exception as e:
            system_prompt = "Bạn là chuyên gia y tế, trả lời rõ ràng, dùng Markdown."

        cleaned_contexts = "\n\n".join([
            ctx.strip()
            .replace("[HEADING]", "### ").replace("[/HEADING]", "\n")
            .replace("[SUBHEADING]", "#### ").replace("[/SUBHEADING]", "\n")
            for ctx in contexts if ctx.strip()
        ])

        try:
            prompt_text = render_template(
                template_name="rag_only.j2",
                system_prompt=system_prompt,
                contexts=cleaned_contexts,
                question=message
            )
        except Exception as e:
            return "Xin lỗi, không thể tạo câu trả lời."

        llm = await self.get_llm_client()
        try:
            response = await self._with_timeout(
                llm.generate(prompt=prompt_text, max_tokens=1800),
                self.LLM_TIMEOUT,
                "RAG Response Generation"
            )
            return self._ensure_markdown(response.strip())
        except asyncio.TimeoutError:
            return "Xin lỗi, tôi đang bận. Vui lòng thử lại sau."
        except Exception as e:
            return "Xin lỗi, tôi đang bận."

    async def _gen_personalized_response(self, message: str, contexts: List[str], user_context: str, user_id: str, first_time: bool = True) -> str:
        profile = await self.get_user_profile(user_id)
        if not profile:
            return "Không tìm thấy hồ sơ người dùng."
        full_name = profile.full_name
        age = profile.age

        try:
            with open("shared/rag_templates/system_prompt.txt", "r", encoding="utf-8") as f:
                system_prompt = f.read().strip()
        except Exception as e:
            system_prompt = "Bạn là bác sĩ nội tiết."

        cleaned_contexts = "\n\n".join([ctx.strip() for ctx in contexts if ctx.strip()])

        try:
            prompt_text = render_template(
                template_name="personalized.j2",
                system_prompt=system_prompt,
                contexts=cleaned_contexts,
                user_context=user_context,
                question=message,
                full_name=full_name,
                age=age,
                first_time=first_time
            )
        except Exception as e:
            return "Xin lỗi, không thể tạo câu trả lời."

        llm = await self.get_llm_client()
        try:
            response = await self._with_timeout(
                llm.generate(prompt=prompt_text, max_tokens=1800),
                self.LLM_TIMEOUT,
                "RAG Response Generation"
            )
            return self._ensure_markdown(response.strip())
        except asyncio.TimeoutError:
            return "Xin lỗi, tôi đang bận. Vui lòng thử lại sau."
        except Exception as e:
            return "Xin lỗi, tôi đang bận."

    async def _analyze_blood_glucose_only(self, user_id: str, question: str, first_time: bool = True, has_previous_trend: bool = False) -> str:
        profile = await self.get_user_profile(user_id)
        if not profile:
            return "Không tìm thấy hồ sơ người dùng."
        records = await self.get_recent_health_records(user_id, "BloodGlucose", top=3)
        if not records:
            return "Chưa có dữ liệu đường huyết để đánh giá."
        values = [r.value for r in records]
        avg = sum(values) / len(values)
        latest = records[0].value
        trend = "tăng" if len(values) >= 2 and values[0] > values[-1] else "giảm" if len(values) >= 2 and values[0] < values[-1] else "ổn định"
        status = "cao" if avg > 8.0 else "trung bình" if avg > 6.0 else "thấp"
        health_summary = f"Đường huyết: trung bình {avg:.1f} mmol/l, gần nhất {latest:.1f} mmol/l — mức {status}, xu hướng {trend}"
        user_context = await self.get_relevant_user_context(user_id, question)

        history_context = ""
        if not first_time:
            if has_previous_trend:
                history_context = "Lần trước, tôi đã phân tích xu hướng cho bác."
            else:
                history_context = "Bác đã hỏi trước đó, nhưng chưa phân tích xu hướng."

        try:
            prompt_text = render_template(
                template_name="trend_response.j2",
                user_context=user_context,
                health_summary=health_summary,
                question=question,
                full_name=profile.full_name,
                age=profile.age,
                first_time=first_time,
                has_previous_trend=has_previous_trend,
                history_context=history_context
            )
        except Exception as e:
            return "Không thể tạo phản hồi chi tiết."
        llm = await self.get_llm_client()
        try:
            response = await self._with_timeout(
                llm.generate(prompt=prompt_text, max_tokens=500),
                self.LLM_TIMEOUT,
                "Health Analysis Response"
            )
            return self._ensure_markdown(response.strip())
        except asyncio.TimeoutError:
            return "Xin lỗi, tôi đang xử lý. Vui lòng thử lại sau."
        except Exception as e:
            return "Xin lỗi, tôi đang xử lý. Vui lòng thử lại sau."

    async def _analyze_blood_pressure_only(self, user_id: str, question: str, first_time: bool = True, has_previous_trend: bool = False) -> str:
        profile = await self.get_user_profile(user_id)
        if not profile:
            return "Không tìm thấy hồ sơ người dùng."
        records = await self.get_recent_health_records(user_id, "BloodPressure", top=3)
        if not records:
            return "Chưa có dữ liệu huyết áp để đánh giá."
        systolic = [r.value for r in records if r.subtype == "tâm thu"]
        if not systolic:
            return "Không có dữ liệu huyết áp tâm thu."
        avg_sys = sum(systolic) / len(systolic)
        avg_dia = sum([r.value for r in records if r.subtype == "tâm trương"]) / len([r for r in records if r.subtype == "tâm trương"])
        if avg_sys > 140 or avg_dia > 90:
            bp_status = "cao – nguy cơ tim mạch tăng"
        elif avg_sys > 120 or avg_dia > 80:
            bp_status = "biên độ cao – cần theo dõi"
        else:
            bp_status = "ổn định"
        health_summary = f"Huyết áp: trung bình {avg_sys:.0f}/{avg_dia:.0f} mmHg — mức {bp_status}"
        user_context = await self.get_relevant_user_context(user_id, question)

        history_context = ""
        if not first_time:
            if has_previous_trend:
                history_context = "Lần trước, tôi đã phân tích huyết áp cho bác."
            else:
                history_context = "Bác đã hỏi trước đó, nhưng chưa phân tích xu hướng."

        try:
            prompt_text = render_template(
                template_name="trend_response.j2",
                user_context=user_context,
                health_summary=health_summary,
                question=question,
                full_name=profile.full_name,
                age=profile.age,
                first_time=first_time,
                has_previous_trend=has_previous_trend,
                history_context=history_context
            )
        except Exception as e:
            return "Không thể tạo phản hồi chi tiết."
        llm = await self.get_llm_client()
        try:
            response = await self._with_timeout(
                llm.generate(prompt=prompt_text, max_tokens=500),
                self.LLM_TIMEOUT,
                "Blood Pressure Analysis"
            )
            return self._ensure_markdown(response.strip())
        except asyncio.TimeoutError:
            return "Xin lỗi, tôi đang xử lý. Vui lòng thử lại."
        except Exception as e:
            return "Xin lỗi, tôi đang xử lý. Vui lòng thử lại."

    async def _analyze_overall_status(self, user_id: str, question: str, first_time: bool = True, has_previous_trend: bool = False) -> str:
        profile = await self.get_user_profile(user_id)
        if not profile:
            return "Không tìm thấy hồ sơ người dùng."
        glucose_records = await self.get_recent_health_records(user_id, "BloodGlucose", top=3)
        bp_records = await self.get_recent_health_records(user_id, "BloodPressure", top=3)
        parts = []
        if glucose_records:
            values = [r.value for r in glucose_records]
            avg = sum(values) / len(values)
            status = "cao" if avg > 8.0 else "trung bình"
            parts.append(f"Đường huyết: trung bình {avg:.1f} mmol/l → mức {status}")
        if bp_records:
            sys = [r.value for r in bp_records if r.subtype == "tâm thu"]
            if sys:
                avg_sys = sum(sys) / len(sys)
                bp_status = "cao" if avg_sys > 140 else "biên độ cao" if avg_sys > 120 else "ổn định"
                parts.append(f"Huyết áp: trung bình {avg_sys:.0f} mmHg → mức {bp_status}")
        if not parts:
            return "Chưa có dữ liệu sức khỏe gần đây để đánh giá."
        health_summary = "\n".join(parts)
        user_context = await self.get_relevant_user_context(user_id, question)

        history_context = ""
        if not first_time:
            if has_previous_trend:
                history_context = "Lần trước, tôi đã tổng hợp tình trạng sức khỏe cho bác."
            else:
                history_context = "Bác đã hỏi trước đó, nhưng chưa phân tích tổng quát."

        try:
            prompt_text = render_template(
                template_name="trend_response.j2",
                user_context=user_context,
                health_summary=health_summary,
                question=question,
                full_name=profile.full_name,
                age=profile.age,
                first_time=first_time,
                has_previous_trend=has_previous_trend,
                history_context=history_context
            )
        except Exception as e:
            return "Không thể tạo phản hồi chi tiết."
        llm = await self.get_llm_client()
        try:
            response = await self._with_timeout(
                llm.generate(prompt=prompt_text, max_tokens=600),
                self.LLM_TIMEOUT,
                "Overall Health Analysis"
            )
            return self._ensure_markdown(response.strip())
        except asyncio.TimeoutError:
            return "Xin lỗi, tôi đang bận. Vui lòng thử lại."
        except Exception as e:
            return "Xin lỗi, tôi đang bận."

    async def generate_health_status_response(
        self,
        user_id: str,
        question: str,
        first_time: bool = True,
        has_previous_trend: bool = False
    ) -> str:
        profile = await self.get_user_profile(user_id)
        if not profile:
            try:
                prompt_text = render_template(
                    template_name="no_profile_response.j2",
                    question=question
                )
            except Exception as e:
                return (
                    "**Tôi hiểu** rằng việc chia sẻ thông tin cá nhân có thể khiến bạn cảm thấy không thoải mái, "
                    "nhưng để tôi hỗ trợ bạn tốt nhất, bạn vui lòng cập nhật một số thông tin cơ bản như tuổi và loại bệnh lý.\n\n"
                    "Chỉ cần vài phút thời gian — sẽ giúp tôi đưa ra lời khuyên **chính xác và phù hợp với hoàn cảnh của bạn**.\n\n"
                    "Bạn không đơn độc trong hành trình này — tôi luôn ở đây để đồng hành."
                )
            llm = await self.get_llm_client()
            try:
                response = await self._with_timeout(
                    llm.generate(prompt=prompt_text, max_tokens=500),
                    self.LLM_TIMEOUT,
                    "No Profile Response"
                )
                return self._ensure_markdown(response.strip())
            except asyncio.TimeoutError:
                return (
                    "**Tôi hiểu** rằng việc chia sẻ thông tin cá nhân có thể khiến bạn cảm thấy không thoải mái, "
                    "nhưng để tôi hỗ trợ bạn tốt nhất, bạn vui lòng cập nhật một số thông tin cơ bản như tuổi và loại bệnh lý.\n\n"
                    "Chỉ cần vài phút thời gian — sẽ giúp tôi đưa ra lời khuyên **chính xác và phù hợp với hoàn cảnh của bạn**.\n\n"
                    "Bạn không đơn độc trong hành trình này — tôi luôn ở đây để đồng hành."
                )
            except Exception as e:
                return (
                    "**Tôi hiểu** rằng việc chia sẻ thông tin cá nhân có thể khiến bạn cảm thấy không thoải mái, "
                    "nhưng để tôi hỗ trợ bạn tốt nhất, bạn vui lòng cập nhật một số thông tin cơ bản như tuổi và loại bệnh lý.\n\n"
                    "Chỉ cần vài phút thời gian — sẽ giúp tôi đưa ra lời khuyên **chính xác và phù hợp với hoàn cảnh của bạn**.\n\n"
                    "Bạn không đơn độc trong hành trình này — tôi luôn ở đây để đồng hành."
                )

        glucose_records = await self.get_recent_health_records(user_id, "BloodGlucose", top=1)
        bp_records = await self.get_recent_health_records(user_id, "BloodPressure", top=1)

        if not glucose_records and not bp_records:
            try:
                prompt_text = render_template(
                    template_name="no_data_response.j2",
                    question=question,
                    full_name=profile.full_name
                )
            except Exception as e:
                return (
                    "**Hiện tôi chưa thấy** có dữ liệu sức khỏe gần đây.\n\n"
                    "Hãy bắt đầu **ghi lại đường huyết 1–2 lần mỗi ngày**."
                )
            llm = await self.get_llm_client()
            try:
                response = await self._with_timeout(
                    llm.generate(prompt=prompt_text, max_tokens=500),
                    self.LLM_TIMEOUT,
                    "No Data Response"
                )
                return self._ensure_markdown(response.strip())
            except asyncio.TimeoutError:
                return (
                    "**Chúng ta chưa có** dữ liệu gần đây.\n\n"
                    "Hãy thử **đo và ghi lại** – tôi sẽ giúp bạn phân tích ngay khi có số liệu."
                )
            except Exception as e:
                return (
                    "**Chúng ta chưa có** dữ liệu gần đây.\n\n"
                    "Hãy thử **đo và ghi lại** – tôi sẽ giúp bạn phân tích ngay khi có số liệu."
                )

        q = question.lower()
        if "huyết áp" in q:
            return await self._analyze_blood_pressure_only(user_id, question, first_time, has_previous_trend)
        elif any(kw in q for kw in ["đường huyết", "glucose"]):
            if not glucose_records:
                try:
                    prompt_text = render_template(
                        template_name="no_glucose_data.j2",
                        question=question,
                        full_name=profile.full_name
                    )
                except Exception as e:
                    return "**Chưa có dữ liệu** đường huyết."
                llm = await self.get_llm_client()
                try:
                    response = await self._with_timeout(
                        llm.generate(prompt=prompt_text, max_tokens=500),
                        self.LLM_TIMEOUT,
                        "No Glucose Data Response"
                    )
                    return self._ensure_markdown(response.strip())
                except asyncio.TimeoutError:
                    return "Hãy bắt đầu đo đường huyết mỗi ngày."
                except Exception as e:
                    return "Hãy bắt đầu đo đường huyết mỗi ngày."
            return await self._analyze_blood_glucose_only(user_id, question, first_time, has_previous_trend)
        else:
            return await self._analyze_overall_status(user_id, question, first_time, has_previous_trend)

    async def get_polite_response_for_invalid_question(self, question: str) -> str:
        try:
            prompt_text = render_template(
                template_name="polite_response.j2",
                question=question
            )
        except Exception as e:
            return (
                "**Tôi hiểu** bạn có thể đang cảm thấy mệt mỏi, nhưng **sức khỏe của bạn rất quan trọng**.\n\n"
                "Hãy tìm sự hỗ trợ từ bác sĩ hoặc người thân – bạn không đơn độc."
            )
        llm = await self.get_llm_client()
        try:
            response = await self._with_timeout(
                llm.generate(prompt=prompt_text, max_tokens=500),
                self.LLM_TIMEOUT,
                "Polite Response"
            )
            return self._ensure_markdown(response.strip())
        except asyncio.TimeoutError:
            return (
                "**Sức khỏe của bạn rất quan trọng**. Hãy tìm sự hỗ trợ — bạn không đơn độc."
            )
        except Exception as e:
            return (
                "**Sức khỏe của bạn rất quan trọng**. Hãy tìm sự hỗ trợ — bạn không đơn độc."
            )

    def _ensure_markdown(self, text: str) -> str:
        if not text.strip():
            return "Tôi chưa thể tạo câu trả lời phù hợp."
        import re
        text = re.sub(r'\*\*(.*?)\*\*', r'**\1**', text)
        text = re.sub(r'\*(.*?)\*', r'*\1*', text)

        if "**" not in text and "*" not in text and "###" not in text:
            if "Xin lỗi" in text:
                text = text.replace("Xin lỗi", "**Xin lỗi**")
            if "cập nhật" in text.lower():
                text = text.replace("cập nhật", "**cập nhật**")
            if "hãy bắt đầu" in text.lower():
                text = text.replace("hãy bắt đầu", "**Hãy bắt đầu**")

        lines = text.split('\n')
        cleaned = []
        in_leak = False
        leak_keywords = ["hãy suy nghĩ", "phân tích", "tôi cần trả lời", "let me think"]
        for line in lines:
            lower_line = line.lower()
            if any(kw in lower_line for kw in leak_keywords):
                in_leak = True
                continue
            if line.startswith("### ") and in_leak:
                in_leak = False
                cleaned.append(line)
            elif not in_leak and line.strip():
                cleaned.append(line)
        result = '\n'.join(cleaned).strip()

        if result and (result.startswith(("I ", "You ")) or "cannot" in result or "Sorry" in result):
            return "Xin lỗi, tôi chỉ hỗ trợ bằng tiếng Việt."
        return result if result else "Tôi chưa có thông tin để trả lời."

    async def execute(self, command: CreateChatCommand) -> Result[None]:
        try:
            # Wrap entire execution with total timeout
            return await self._with_timeout(
                self._execute_internal(command),
                self.TOTAL_TIMEOUT,
                "Complete Chat Processing"
            )
        except asyncio.TimeoutError as e:
            self.logger.error(f"Total execution timeout: {e}")
            return Result.failure(
                code=ChatMessage.CHAT_CREATED_FAILED.code,
                message="Xin lỗi, xử lý quá lâu. Vui lòng thử lại."
            )
        except Exception as e:
            self.logger.error(f"Error in _execute_internal: {e}", exc_info=True)
            return Result.failure(
                code=ChatMessage.CHAT_CREATED_FAILED.code,
                message=ChatMessage.CHAT_CREATED_FAILED.message
            )
    
    async def _execute_internal(self, command: CreateChatCommand) -> Result[None]:
        try:
            settings_doc = await self.db.settings.find_one({})
            if not settings_doc:
                return Result.failure(
                    code=SettingMessage.NOT_FOUND.code,
                    message=SettingMessage.NOT_FOUND.message
                )
            settings = SettingModel.from_dict(settings_doc)

            session = await self.create_session(
                user_id=command.user_id,
                title=command.content,
                session_id=command.session_id
            )
            if not session:
                return Result.failure(message="Không tạo được session.")

            user_chat = ChatHistoryModel(
                session_id=str(session.id),
                user_id=command.user_id,
                content=command.content,
                role=ChatRoleType.USER
            )
            await self.save_data(user_chat)

            histories = await self.get_histories(session.id)
            histories.reverse()

            ai_messages = [msg for msg in histories if msg.role == ChatRoleType.AI]
            first_time = len(ai_messages) == 0

            # Không dùng has_analyzed_trend
            has_previous_trend = any(
                "xu hướng" in msg.content.lower() or
                "gần đây" in msg.content.lower() or
                "đánh giá" in msg.content.lower() or
                "phân tích" in msg.content.lower()
                for msg in ai_messages
            )

            content_lower = command.content.lower()

            rag_safe_keywords = [
                "đường huyết sau ăn", "người tiểu đường nên ăn gì", "biến chứng tiểu đường",
                "cách tiêm insulin", "chỉ số hba1c", "đái tháo đường là gì",
                "insulin hoạt động bao lâu", "ăn gì khi bị tiểu đường"
            ]

            sensitive_keywords = [
                "chết", "bỏ thuốc", "ăn đường cả ngày", "không cần kiểm soát",
                "làm sao để", "có nên bỏ", "tôi mệt quá"
            ]

            if any(kw in content_lower for kw in rag_safe_keywords):
                question_type = "rag_only"
            elif any(kw in content_lower for kw in sensitive_keywords):
                question_type = "invalid"
            else:
                question_type = await self.classify_question_type(command.content)

            gen_text = ""

            if question_type == "invalid":
                gen_text = await self.get_polite_response_for_invalid_question(command.content)
            elif question_type == "trend":
                gen_text = await self.generate_health_status_response(command.user_id, command.content, first_time, has_previous_trend)
            elif question_type == "rag_only":
                context_texts = await self._retrieve_rag_context(command.content, histories, settings)
                gen_text = await self._gen_rag_only_response(command.content, context_texts)
            elif question_type == "personal":
                context_texts = await self._retrieve_rag_context(command.content, histories, settings)
                user_context = await self.get_relevant_user_context(command.user_id, command.content)
                if context_texts and user_context:
                    gen_text = await self._gen_personalized_response(command.content, context_texts, user_context, command.user_id, first_time)
                elif context_texts:
                    gen_text = await self._gen_rag_only_response(command.content, context_texts)
                else:
                    gen_text = await self.generate_health_status_response(command.user_id, command.content, first_time, has_previous_trend)
            else:
                context_texts = await self._retrieve_rag_context(command.content, histories, settings)
                gen_text = await self._gen_rag_only_response(command.content, context_texts)

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
            self.logger.error(f"Error in _execute_internal: {e}", exc_info=True)
            return Result.failure(
                code=ChatMessage.CHAT_CREATED_FAILED.code,
                message=ChatMessage.CHAT_CREATED_FAILED.message
            )