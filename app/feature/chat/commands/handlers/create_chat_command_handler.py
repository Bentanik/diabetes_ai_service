import os
import dotenv
import asyncio
from datetime import datetime
from typing import List, Optional, Dict
from bson import ObjectId
from tenacity import retry, stop_after_attempt, wait_exponential

from core.cqrs import CommandRegistry, CommandHandler, Mediator
from core.llm import QwenLLM
from core.result import Result
from shared.messages import ChatMessage, SettingMessage
from app.feature.train_ai import GetRetrievedContextQuery
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
    """Handler for processing CreateChatCommand to manage chat sessions and responses."""
    def __init__(self):
        super().__init__()
        self.logger = get_logger(__name__)
        self.db = get_collections()
        self.llm_client = None
        self.retriever_cache = {}
        self.LLM_TIMEOUT = 75  # Increased for complex queries
        self.DB_TIMEOUT = 15
        self.TOTAL_TIMEOUT = 120

    async def get_llm_client(self) -> QwenLLM:
        """Initialize or return cached QwenLLM client."""
        if self.llm_client is None:
            self.llm_client = QwenLLM(
                model=os.getenv("QWEN_MODEL", "qwen2.5:3b-instruct"),
                base_url=os.getenv("QWEN_URL", "http://localhost:11434")
            )
        return self.llm_client

    async def _with_timeout(self, coro, timeout_seconds: int, operation_name: str):
        """Execute coroutine with timeout and logging."""
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

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
    async def create_session(
        self,
        user_id: str,
        title: str,
        session_id: Optional[str] = None
    ) -> Optional[ChatSessionModel]:
        """Create or retrieve a chat session."""
        if not user_id or not title:
            self.logger.error("Invalid user_id or title")
            return None
        try:
            if user_id == "admin":
                doc = await self._with_timeout(
                    self.db.chat_sessions.find_one({"user_id": "admin"}),
                    self.DB_TIMEOUT,
                    "Find Admin Session"
                )
                if doc:
                    return ChatSessionModel.from_dict(doc)
                session = ChatSessionModel(user_id="admin", title="Test AI")
                result = await self._with_timeout(
                    self.db.chat_sessions.insert_one(session.to_dict()),
                    self.DB_TIMEOUT,
                    "Insert Admin Session"
                )
                session._id = result.inserted_id
                return session

            if session_id:
                try:
                    obj_id = ObjectId(session_id)
                    doc = await self._with_timeout(
                        self.db.chat_sessions.find_one({"_id": obj_id}),
                        self.DB_TIMEOUT,
                        "Find Session by ID"
                    )
                    if doc:
                        return ChatSessionModel.from_dict(doc)
                except ValueError:
                    self.logger.error(f"Invalid session_id: {session_id}")
                    return None

            session_title = title[:100] + "..." if len(title) > 100 else title
            session = ChatSessionModel(user_id=user_id, title=session_title)
            result = await self._with_timeout(
                self.db.chat_sessions.insert_one(session.to_dict()),
                self.DB_TIMEOUT,
                "Insert New Session"
            )
            session._id = result.inserted_id
            return session

        except Exception as e:
            self.logger.error(f"Error creating session: {e}", exc_info=True)
            return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
    async def update_session(self, session_id: str) -> bool:
        """Update session's last modified timestamp."""
        try:
            obj_id = ObjectId(session_id)
            result = await self._with_timeout(
                self.db.chat_sessions.update_one(
                    {"_id": obj_id},
                    {"$set": {"updated_at": datetime.utcnow()}}
                ),
                self.DB_TIMEOUT,
                "Update Session"
            )
            return result.modified_count > 0
        except (ValueError, Exception) as e:
            self.logger.error(f"Update session failed: {e}", exc_info=True)
            return False

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
    async def get_histories(self, session_id: str) -> List[ChatHistoryModel]:
        """Retrieve recent chat history for a session."""
        try:
            obj_id = ObjectId(session_id)
            cursor = self.db.chat_histories.find({"session_id": str(obj_id)}) \
                .sort("updated_at", -1).limit(20)
            docs = await self._with_timeout(
                cursor.to_list(length=20),
                self.DB_TIMEOUT,
                "Get Chat Histories"
            )
            histories = []
            for doc in docs:
                model = ChatHistoryModel.from_dict(doc)
                if isinstance(model.role, str):
                    model.role = ChatRoleType.USER if model.role.lower() == "user" else ChatRoleType.AI
                histories.append(model)
            return histories
        except (ValueError, Exception) as e:
            self.logger.error(f"Cannot get chat history: {e}", exc_info=True)
            return []

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
    async def save_data(self, data: ChatHistoryModel) -> bool:
        """Save chat history to database."""
        if not data or not data.session_id or not data.content:
            self.logger.error("Invalid chat history data")
            return False
        try:
            if isinstance(data.session_id, ObjectId):
                data.session_id = str(data.session_id)
            result = await self._with_timeout(
                self.db.chat_histories.insert_one(data.to_dict()),
                self.DB_TIMEOUT,
                "Save Chat History"
            )
            return result.acknowledged
        except Exception as e:
            self.logger.error(f"Save chat history failed: {e}", exc_info=True)
            return False

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
    async def get_user_profile(self, user_id: str) -> Optional[UserProfileModel]:
        """Retrieve user profile from database."""
        if not user_id:
            self.logger.error("Invalid user_id")
            return None
        try:
            doc = await self._with_timeout(
                self.db.user_profiles.find_one({"user_id": user_id}),
                self.DB_TIMEOUT,
                "Get User Profile"
            )
            return UserProfileModel.from_dict(doc) if doc else None
        except Exception as e:
            self.logger.error(f"Không thể lấy hồ sơ người dùng {user_id}: {e}")
            return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
    async def get_recent_health_records(self, user_id: str, record_type: str, top: int = 3) -> List[HealthRecordModel]:
        """Retrieve recent health records for a user."""
        if not user_id or not record_type:
            self.logger.error(f"Invalid user_id or record_type: {user_id}, {record_type}")
            return []
        try:
            cursor = self.db.health_records.find({
                "user_id": user_id,
                "type": record_type
            }).sort("timestamp", -1).limit(top)
            docs = await self._with_timeout(
                cursor.to_list(length=top),
                self.DB_TIMEOUT,
                f"Get {record_type} Records"
            )
            return [HealthRecordModel.from_dict(doc) for doc in docs if doc]
        except Exception as e:
            self.logger.error(f"Lỗi lấy chỉ số {record_type}: {e}")
            return []

    async def get_relevant_user_context(self, user_id: str, question: str) -> str:
        """Generate relevant user context based on question."""
        profile = await self.get_user_profile(user_id)
        if not profile or user_id == "admin":
            return "Bác ơi, tui chưa có thông tin về bác. Hãy cập nhật hồ sơ để tui hỗ trợ nhé!"

        parts = [
            f"Bác {profile.full_name} (ID: {profile.patient_id}), {profile.age} tuổi, {profile.gender}, "
            f"đang quản lý tiểu đường loại {profile.diabetes_type}."
        ]
        q = question.lower()

        if any(kw in q for kw in ["đường huyết", "glucose", "chỉ số đường"]):
            records = await self.get_recent_health_records(user_id, "Đường huyết", top=3)
            if records:
                summary = ", ".join([f"{r.value:.1f} mmol/L ({r.timestamp.strftime('%d/%m')})" for r in records])
                parts.append(f"Đường huyết gần đây: {summary}.")
            else:
                parts.append("Đường huyết: chưa có dữ liệu gần đây.")
            if profile.complications:
                parts.append(f"Biến chứng hiện tại: {', '.join(profile.complications)}.")

        if any(kw in q for kw in ["huyết áp", "blood pressure", "tim mạch"]):
            records = await self.get_recent_health_records(user_id, "Huyết áp", top=3)
            if records:
                sys = [r.value for r in records if r.subtype == "Tâm thu"]
                dia = [r.value for r in records if r.subtype == "Tâm trương"]
                if sys and dia:
                    parts.append(f"Huyết áp gần đây: trung bình {sum(sys)/len(sys):.0f}/{sum(dia)/len(dia):.0f} mmHg.")
                else:
                    parts.append("Huyết áp: dữ liệu không đầy đủ.")
            else:
                parts.append("Huyết áp: chưa có dữ liệu gần đây.")

        if any(kw in q for kw in ["insulin", "tiêm"]):
            if profile.insulin_schedule:
                parts.append(f"Lịch tiêm insulin: {profile.insulin_schedule}.")

        if any(kw in q for kw in ["ăn", "chế độ", "lối sống"]):
            if profile.lifestyle:
                parts.append(f"Lối sống: {profile.lifestyle}.")
            if profile.bmi:
                parts.append(f"BMI: {profile.bmi:.1f}.")

        return "\n".join(parts) or "Tui cần thêm thông tin để trả lời chính xác hơn, bác chia sẻ thêm nhé!"

    async def classify_question_type(self, question: str, histories: List[ChatHistoryModel]) -> Dict[str, any]:
        """Classify question type using LLM with context from history."""
        llm = await self.get_llm_client()
        history_text = "\n".join([
            f"- {msg.role}: {msg.content}"
            for msg in histories[-3:] if msg.content
        ]) if histories else "Không có lịch sử trò chuyện."

        prompt = f"""
Bạn là hệ thống phân loại câu hỏi y tế chuyên về bệnh tiểu đường và sức khỏe liên quan. 
Nhiệm vụ: Phân loại câu hỏi thành **chính xác 1 loại** từ các loại sau:
- `greeting`: Lời chào như "xin chào", "chào bạn", "hello".
- `invalid`: Câu hỏi tiêu cực, tự tử, bỏ điều trị ("chết", "bỏ thuốc", "mệt quá").
- `personal_info`: Hỏi về chỉ số, thuốc, biến chứng, chế độ ăn của bản thân ("của tôi", "tình trạng tôi").
- `trend_analysis`: Hỏi về xu hướng, chỉ số gần đây ("gần đây", "xu hướng", "có ổn không").
- `relational`: So sánh tiểu đường với bệnh khác ("ung thư", "trầm cảm").
- `rag_only`: Kiến thức chung về bệnh, nguyên nhân, loại bệnh ("có mấy loại", "là gì").

**Trả về**: Chỉ **1 từ** (greeting, invalid, personal_info, trend_analysis, relational, rag_only), không giải thích, không viết hoa.

**Lưu ý**:
- Dùng lịch sử trò chuyện để hiểu ngữ cảnh.
- Ưu tiên `rag_only` cho câu hỏi kiến thức chung về tiểu đường.
- Không nhầm "người tiểu đường" với "personal_info" nếu không có chia sẻ cá nhân.
- Câu hỏi về "chỉ số gần đây" hoặc "có ổn không" thuộc `trend_analysis`.

**Ví dụ**:
- "Bệnh tiểu đường có mấy loại vậy" → rag_only
- "Vậy còn các chỉ số gần đây tui có ổn không" → trend_analysis
- "Bệnh ung thư có mấy loại vậy" → rag_only
- "Đường huyết của tôi thế nào" → personal_info
- "Chào bạn" → greeting
- "Tôi mệt quá, sống làm gì" → invalid
- "Tiểu đường liên quan gì đến ung thư" → relational

**Câu hỏi hiện tại**: "{question}"
**Lịch sử trò chuyện**: 
{history_text}

**Loại**:
""".strip()

        try:
            response = await self._with_timeout(
                llm.generate(prompt=prompt, max_tokens=20, temperature=0.01),
                self.LLM_TIMEOUT,
                "Question Classification"
            )
            response = response.strip().lower()
            valid_types = ["greeting", "invalid", "personal_info", "trend_analysis", "relational", "rag_only"]
            return {"type": response} if response in valid_types else {"type": "rag_only"}
        except Exception as e:
            self.logger.error(f"LLM classification failed: {e}")
            q = question.lower()
            if any(kw in q for kw in ["chết", "bỏ thuốc", "mệt quá"]):
                return {"type": "invalid"}
            if any(kw in q for kw in ["chào", "hello", "hi "]):
                return {"type": "greeting"}
            if any(kw in q for kw in ["gần đây", "xu hướng", "có ổn không"]):
                return {"type": "trend_analysis"}
            if any(kw in q for kw in ["so với", "liên quan", "ung thư", "trầm cảm"]):
                return {"type": "relational"}
            if any(phrase in q for phrase in ["của tôi", "tình trạng tôi", "lịch tiêm của tôi"]):
                return {"type": "personal_info"}
            return {"type": "rag_only"}

    def _is_valid_context(self, content: str) -> bool:
        """Validate context content to avoid code or irrelevant data."""
        text = content.strip().lower()
        if len(text) < 30:
            return False
        suspicious = ["import ", "def ", "class ", "from ", "os.", "dotenv", "error", "exception", "traceback"]
        return not any(s in text for s in suspicious)

    async def _retrieve_rag_context(self, query: str, settings: SettingModel) -> List[str]:
        """Retrieve relevant context from vector store."""
        if not settings.list_knowledge_ids:
            self.logger.info("No knowledge IDs configured for RAG")
            return []
        try:
            retrieval_result: Result = await self._with_timeout(
                Mediator.send(GetRetrievedContextQuery(query=query)),
                30,
                "RAG Retrieval"
            )
            if retrieval_result.is_success and retrieval_result.data:
                contexts = [
                    dto.content for dto in retrieval_result.data
                    if dto.content and self._is_valid_context(dto.content)
                ]
                self.logger.debug(f"Found {len(contexts)} valid contexts for query: '{query}'")
                return contexts[:settings.top_k]
            self.logger.info(f"Không tìm thấy tài liệu đủ liên quan cho: '{query}'")
            return []
        except Exception as e:
            self.logger.error(f"RAG retrieval failed: {e}")
            return []

    async def _gen_rag_only_response(self, message: str, contexts: List[str], histories: List[ChatHistoryModel]) -> str:
        """Generate response for general knowledge questions."""
        if not contexts:
            q = message.lower()
            if any(kw in q for kw in ["có mấy loại", "ngoài loại 1 và 2"]):
                return (
                    "**Bệnh tiểu đường có các loại chính**:\n\n"
                    "1. **Loại 1**: Cơ thể không sản xuất insulin do hệ miễn dịch tấn công tế bào beta.\n"
                    "2. **Loại 2**: Cơ thể kháng insulin hoặc không sản xuất đủ insulin.\n"
                    "3. **Tiểu đường thai kỳ**: Xảy ra trong thai kỳ, thường tự hết sau sinh.\n"
                    "4. **MODY và các loại hiếm**: Do gen, ít gặp.\n\n"
                    "Loại 1 và loại 2 chiếm phần lớn (~95%) các trường hợp. Bác muốn biết thêm về loại nào không?"
                )
            if "ung thư" in q:
                return (
                    "Bác ơi, hiện tui chỉ chuyên về **bệnh tiểu đường** và các vấn đề liên quan như đường huyết, insulin, chế độ ăn. "
                    "Về ung thư, tui chưa có đủ thông tin chính xác để trả lời. "
                    "Bác có thể hỏi về **tiểu đường** hoặc chia sẻ thêm để tui hỗ trợ nha!"
                )
            return (
                "Bác ơi, tui chưa tìm thấy thông tin phù hợp cho câu hỏi này. 😅\n\n"
                "Hãy hỏi về **đường huyết, insulin, chế độ ăn uống** hoặc bệnh tiểu đường, tui sẽ trả lời ngay!"
            )

        try:
            with open("shared/rag_templates/system_prompt.txt", "r", encoding="utf-8") as f:
                system_prompt = f.read().strip()
        except Exception:
            system_prompt = "Bạn là chuyên gia y tế về bệnh tiểu đường, trả lời rõ ràng, thân thiện bằng tiếng Việt, dùng Markdown."

        full_context = "\n\n---\n\n".join([
            f"[TÀI LIỆU {i+1}]\n{ctx.strip()}" for i, ctx in enumerate(contexts)
        ]) if contexts else "Không có tài liệu liên quan."

        try:
            prompt_text = render_template(
                template_name="rag_only.j2",
                system_prompt=system_prompt,
                contexts=full_context,
                question=message,
                histories=histories
            )
        except Exception as e:
            self.logger.error(f"Template rendering failed: {e}")
            return "Ôi, tui gặp chút trục trặc khi tạo câu trả lời. Bác hỏi lại nha!"

        llm = await self.get_llm_client()
        try:
            response = await self._with_timeout(
                llm.generate(prompt=prompt_text, max_tokens=400, temperature=0.7),
                self.LLM_TIMEOUT,
                "RAG Response Generation"
            )
            return self._ensure_markdown(response.strip())
        except asyncio.TimeoutError:
            return "Ôi, tui bị kẹt chút rồi, bác thử hỏi lại nha!"
        except Exception as e:
            self.logger.error(f"LLM generation failed: {e}")
            return "Tui gặp vấn đề nhỏ, bác hỏi lại để tui hỗ trợ tiếp nhé!"

    async def _gen_personalized_response(self, message: str, contexts: List[str], user_context: str, user_id: str, first_time: bool, histories: List[ChatHistoryModel]) -> str:
        """Generate personalized response based on user context."""
        profile = await self.get_user_profile(user_id)
        if not profile:
            return "Bác ơi, tui chưa có thông tin hồ sơ của bác. Hãy cập nhật để tui hỗ trợ nha!"

        try:
            with open("shared/rag_templates/system_prompt.txt", "r", encoding="utf-8") as f:
                system_prompt = f.read().strip()
        except Exception:
            system_prompt = "Bạn là bác sĩ nội tiết, trả lời thân thiện, rõ ràng bằng tiếng Việt, dùng Markdown."

        full_context = "\n\n---\n\n".join([
            f"[TÀI LIỆU {i+1}]\n{ctx.strip()}" for i, ctx in enumerate(contexts)
        ]) if contexts else "Không có tài liệu liên quan."

        history_context = ""
        if not first_time:
            history_context = (
                "Lần trước tui đã xem qua tình trạng của bác rồi. " if any("xu hướng" in msg.content.lower() for msg in histories if msg.role == ChatRoleType.AI)
                else "Bác đã hỏi tui trước đó, giờ tui sẽ trả lời chi tiết hơn nha."
            )

        try:
            prompt_text = render_template(
                template_name="personalized.j2",
                system_prompt=system_prompt,
                contexts=full_context,
                user_context=user_context,
                question=message,
                full_name=profile.full_name,
                age=profile.age,
                first_time=first_time,
                history_context=history_context,
                histories=histories
            )
        except Exception as e:
            self.logger.error(f"Template personalized.j2 rendering failed: {e}")
            return "Ôi, tui gặp chút trục trặc khi tạo câu trả lời. Bác hỏi lại nha!"

        llm = await self.get_llm_client()
        try:
            response = await self._with_timeout(
                llm.generate(prompt=prompt_text, max_tokens=500, temperature=0.7),
                self.LLM_TIMEOUT,
                "Personalized Response Generation"
            )
            return self._ensure_markdown(response.strip())
        except asyncio.TimeoutError:
            return f"Chào {'bác' if profile.age >= 50 else 'anh/chị'} {profile.full_name}, tui đang xử lý chậm chút. Bác hỏi lại nha!"
        except Exception as e:
            self.logger.error(f"LLM generation failed: {e}")
            return "Tui gặp vấn đề nhỏ, bác hỏi lại để tui hỗ trợ tiếp nhé!"

    async def get_polite_response_for_invalid_question(self, question: str) -> str:
        """Generate polite response for invalid questions."""
        try:
            prompt_text = render_template(
                template_name="polite_response.j2",
                question=question
            )
        except Exception:
            return (
                "Bác ơi, tui hiểu bác có thể đang mệt mỏi, nhưng **sức khỏe rất quan trọng**! 😊\n\n"
                "Hãy chia sẻ thêm về tình trạng của bác hoặc hỏi về **đường huyết, insulin**, tui sẽ hỗ trợ ngay. "
                "Nếu cần, bác nên gặp bác sĩ hoặc người thân để được giúp đỡ thêm nha!"
            )
        llm = await self.get_llm_client()
        try:
            response = await self._with_timeout(
                llm.generate(prompt=prompt_text, max_tokens=300, temperature=0.7),
                self.LLM_TIMEOUT,
                "Polite Response"
            )
            return self._ensure_markdown(response.strip())
        except asyncio.TimeoutError:
            return (
                "Bác ơi, sức khỏe của bác rất quan trọng! Hãy tìm hỗ trợ từ bác sĩ hoặc người thân nha, tui luôn ở đây để giúp!"
            )
        except Exception:
            return (
                "Bác ơi, sức khỏe của bác rất quan trọng! Hãy tìm hỗ trợ từ bác sĩ hoặc người thân nha, tui luôn ở đây để giúp!"
            )

    def _ensure_markdown(self, text: str) -> str:
        """Ensure response is clean and in valid Markdown format."""
        if not text.strip():
            return "Bác ơi, tui chưa tìm ra câu trả lời phù hợp. Hỏi lại nha!"
        
        lower_text = text.lower()
        if any(kw in lower_text for kw in ["import ", "def ", "class ", "from ", "os.", "dotenv", "bài trả lời", "tuân thủ"]):
            return (
                "Tui chưa có thông tin về câu hỏi này. 😅\n\n"
                "Bác hỏi về đường huyết, huyết áp, hay chế độ ăn uống đi, tui sẽ trả lời ngay!"
            )

        import re
        text = re.sub(r'\*\*(.*?)\*\*', r'**\1**', text)
        text = re.sub(r'\*(.*?)\*', r'*\1*', text)

        lines = text.split('\n')
        cleaned = []
        in_leak = False
        leak_keywords = ["hãy suy nghĩ", "phân tích", "tôi cần trả lời", "let me think", "step by step", "bài trả lời", "tuân thủ"]
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

        if result and (result.startswith(("I ", "You ")) or "cannot" in result.lower() or "sorry" in result.lower()):
            return "Xin lỗi, tui chỉ hỗ trợ bằng tiếng Việt, bác hỏi lại nha!"
        return result if result else "Tui chưa có thông tin để trả lời, bác hỏi thêm chi tiết nha!"

    async def execute(self, command: CreateChatCommand) -> Result[None]:
        """Execute the CreateChatCommand to process user query and generate response."""
        if not command or not command.user_id or not command.content:
            self.logger.error("Invalid command data")
            return Result.failure(
                code=ChatMessage.CHAT_CREATED_FAILED.code,
                message="Dữ liệu không hợp lệ, vui lòng kiểm tra lại."
            )
        try:
            return await self._with_timeout(
                self._execute_internal(command),
                self.TOTAL_TIMEOUT,
                "Complete Chat Processing"
            )
        except asyncio.TimeoutError as e:
            self.logger.error(f"Total execution timeout: {e}")
            return Result.failure(
                code=ChatMessage.CHAT_CREATED_FAILED.code,
                message="Ôi, tui xử lý hơi lâu, bác thử lại nha!"
            )
        except Exception as e:
            self.logger.error(f"Error in _execute_internal: {e}", exc_info=True)
            return Result.failure(
                code=ChatMessage.CHAT_CREATED_FAILED.code,
                message="Tui gặp vấn đề nhỏ, bác thử lại nha!"
            )

    async def _execute_internal(self, command: CreateChatCommand) -> Result[None]:
        """Internal execution logic for chat command."""
        try:
            settings_doc = await self._with_timeout(
                self.db.settings.find_one({}),
                self.DB_TIMEOUT,
                "Get Settings"
            )
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
            if not await self.save_data(user_chat):
                return Result.failure(message="Không lưu được tin nhắn người dùng.")

            histories = await self.get_histories(session.id)
            histories.reverse()

            ai_messages = [msg for msg in histories if msg.role == ChatRoleType.AI]
            first_time = len(ai_messages) == 0
            has_previous_trend = any(
                kw in msg.content.lower()
                for kw in ["xu hướng", "gần đây", "đánh giá", "phân tích", "thay đổi"]
                for msg in ai_messages
            )

            contexts = await self._retrieve_rag_context(command.content, settings)
            self.logger.info(f"RAG Retrieval: found {len(contexts)} contexts")

            classification = await self.classify_question_type(command.content, histories)
            question_type = classification["type"]
            self.logger.info(f"🔍 Question: '{command.content}' → Type: {question_type}")

            gen_text = ""
            if question_type == "greeting":
                profile = await self.get_user_profile(command.user_id)
                name = profile.full_name if profile else "bạn"
                gen_text = (
                    f"Chào {'bác' if profile and profile.age >= 50 else 'anh/chị'} {name}! 😊\n\n"
                    "Tui là trợ lý y tế chuyên về tiểu đường. Bác muốn hỏi gì hôm nay? Ví dụ như đường huyết, insulin, hay chế độ ăn uống nè!"
                )
            elif question_type == "invalid":
                gen_text = await self.get_polite_response_for_invalid_question(command.content)
            elif question_type == "trend_analysis":
                gen_text = await self._gen_personalized_response(
                    message=command.content,
                    contexts=contexts,
                    user_context=await self.get_relevant_user_context(command.user_id, command.content),
                    user_id=command.user_id,
                    first_time=first_time,
                    histories=histories
                )
            elif question_type == "personal_info":
                gen_text = await self._gen_personalized_response(
                    message=command.content,
                    contexts=contexts,
                    user_context=await self.get_relevant_user_context(command.user_id, command.content),
                    user_id=command.user_id,
                    first_time=first_time,
                    histories=histories
                )
            elif question_type == "relational":
                gen_text = await self._gen_rag_only_response(command.content, contexts, histories)
            else:
                gen_text = await self._gen_rag_only_response(command.content, contexts, histories)

            ai_chat = ChatHistoryModel(
                session_id=str(session.id),
                user_id=command.user_id,
                content=gen_text,
                role=ChatRoleType.AI
            )
            if not await self.save_data(ai_chat):
                return Result.failure(message="Không lưu được câu trả lời AI.")

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
                message="Tui gặp vấn đề nhỏ, bác thử lại nha!"
            )