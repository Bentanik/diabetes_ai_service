"""
Chat Command Handler - Xử lý lệnh trò chuyện

File này định nghĩa handler để xử lý ChatCommand, thực hiện việc
trò chuyện với AI sử dụng RAG (Retrieval-Augmented Generation).
"""

from typing import List, Optional
from bson import ObjectId
from app.database.enums import ChatRoleType
from app.database.models import ChatHistoryModel
from app.database.models.setting_model import SettingModel
from core.cqrs import CommandHandler
from core.llm.gemini.config import GeminiConfig
from rag.vector_store.operations import VectorStoreOperations
from shared.messages import SessionChatResult
from shared.messages.setting_message import SettingResult
from ..create_chat_command import CreateChatCommand
from core.cqrs import CommandRegistry
from core.result import Result
from shared.messages import ChatResult
from utils import get_logger
from app.database import get_collections
from app.database.models import ChatSessionModel
from core.llm.gemini.manager import GeminiChatManager
from core.llm.gemini.schemas import Message, Role
from app.dto.models import ChatHistoryModelDTO


class DiabetesRAGPrompt:
    def __init__(self):
        self.base_system_config = """
Bạn là một chuyên gia y tế thân thiện và chuyên nghiệp chuyên về bệnh tiểu đường.

Các nguyên tắc cơ bản:
- Trả lời bằng tiếng Việt một cách rõ ràng, dễ hiểu và thân thiện
- Tránh sử dụng thuật ngữ y khoa khó hiểu, giải thích các thuật ngữ bằng ngôn ngữ đơn giản
- Chỉ trả lời những gì đã có trong tài liệu
- Thể hiện sự đồng cảm và đưa ra lời khuyên thiết thực
- Trả lời ngắn gọn tầm 500 từ trở xuống
- Khuyến khích người dùng tham khảo ý kiến chuyên gia y tế khi cần thiết
- Dựa trên kiến thức chuyên môn và thông tin tham khảo được cung cấp
""".strip()

    def build_complete_system_prompt(
        self, 
        context: Optional[str] = None,
        custom_system_prompt: Optional[str] = None,
        context_prompt: Optional[str] = None
    ) -> str:
        if not context or not context.strip():
            prompt_no_context = self.base_system_config + "\n\n" + \
                "Hiện tại tôi không tìm thấy tài liệu phù hợp để trả lời câu hỏi của bạn.\n" \
                "Bạn có thể thử hỏi lại hoặc tham khảo ý kiến chuyên gia y tế."
            if custom_system_prompt and custom_system_prompt.strip():
                prompt_no_context += "\n" + custom_system_prompt.strip()
            return prompt_no_context

        prompt_parts = [self.base_system_config]
        if custom_system_prompt and custom_system_prompt.strip():
            prompt_parts.append("\n" + custom_system_prompt.strip())

        if context_prompt and context_prompt.strip():
            if "{context}" in context_prompt:
                context_section = context_prompt.format(context=context.strip())
            else:
                context_section = f"{context_prompt}\n{context.strip()}"
        else:
            context_section = context.strip()

        if context_section:
            prompt_parts.append("\n" + context_section)

        return "\n".join(prompt_parts)


@CommandRegistry.register_handler(CreateChatCommand)
class CreateChatCommandHandler(CommandHandler):

    def __init__(self):
        super().__init__()
        self.db = get_collections()
        self.logger = get_logger(__name__)
        self.vector_operations = VectorStoreOperations.get_instance()
        self.prompt_builder = DiabetesRAGPrompt()

    async def execute(self, command: CreateChatCommand) -> Result[None]:
        try:
            session_id = "session_admin"
            if command.user_id == "admin":
                chat_user = ChatHistoryModel(
                    session_id=session_id,
                    user_id=command.user_id,
                    content=command.content,
                    role=ChatRoleType.USER,
                )
                await self.db.chat_histories.insert_one(chat_user.to_dict())
            else:
                if command.session_id and not ObjectId.is_valid(command.session_id):
                    return Result.failure(
                        message="Phiên trò chuyện không hợp lệ",
                        code="error",
                    )
                if command.session_id:
                    is_session_exists = await self.db.chat_sessions.count_documents(
                        {"_id": ObjectId(command.session_id)}
                    )
                    if not is_session_exists:
                        return Result.failure(
                            message=SessionChatResult.SESSION_NOT_FOUND.message,
                            code=SessionChatResult.SESSION_NOT_FOUND.code,
                        )
                session_id = await self._create_session(command)
                chat_user = ChatHistoryModel(
                    session_id=session_id,
                    user_id=command.user_id,
                    content=command.content,
                    role=ChatRoleType.USER,
                )
                await self.db.chat_histories.insert_one(chat_user.to_dict())

            setting = await self.db.settings.find_one({})
            if not setting:
                return Result.failure(
                    message=SettingResult.NOT_FOUND.message,
                    code=SettingResult.NOT_FOUND.code,
                    data=[],
                )
            setting = SettingModel.from_dict(setting)

            context = None
            try:
                retrieved_documents = await self.vector_operations.search(
                    query_text=command.content,
                    top_k=setting.top_k,
                    score_threshold=setting.search_accuracy,
                    collection_names=setting.list_knowledge_ids if setting.list_knowledge_ids else None,
                )
                context = self._combine_retrieved_context(retrieved_documents)
                self.logger.info(f"Retrieved {len(retrieved_documents)} documents for RAG context")
            except Exception as e:
                self.logger.warning(f"Lỗi khi tìm kiếm RAG context: {e}")
                context = None

            ai_response = await self.process_chat_with_ai(
                session_id=session_id,
                user_question=command.content,
                context=context,
                user_id=command.user_id,
                setting=setting,
            )

            if ai_response.is_success:
                return Result.success(
                    message=ChatResult.CHAT_CREATED.message,
                    code=ChatResult.CHAT_CREATED.code,
                    data=ai_response.data,
                )
            else:
                return ai_response

        except Exception as e:
            self.logger.error(f"Lỗi khi tạo cuộc trò chuyện: {e}", exc_info=True)
            return Result.failure(message="Lỗi hệ thống", code="error")

    def _combine_retrieved_context(self, documents: List) -> str:
        if not documents:
            return ""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            content = doc.get("content", "") if isinstance(doc, dict) else str(doc)
            if content.strip():
                context_parts.append(f"[Tài liệu {i}]: {content.strip()}")
        return "\n\n".join(context_parts)

    async def _create_session(self, command: CreateChatCommand) -> str:
        try:
            if not command.session_id:
                session = ChatSessionModel(
                    user_id=command.user_id,
                    title=(command.content[:100] + "..." if len(command.content) > 100 else command.content),
                )
                await self.db.chat_sessions.insert_one(session.to_dict())
                return str(session.id)
            await self.db.chat_sessions.update_one(
                {"_id": ObjectId(command.session_id)},
                {"$set": {"updated_at": command.updated_at}}
            )
            return command.session_id
        except Exception as e:
            self.logger.error(f"Lỗi khi tạo phiên trò chuyện: {e}", exc_info=True)
            raise

    def _convert_chat_history_to_messages(self, chat_histories: List[ChatHistoryModel]) -> List[Message]:
        messages = []
        for chat in chat_histories:
            if chat.role == ChatRoleType.USER:
                messages.append(Message(role=Role.USER, content=chat.content))
            elif chat.role == ChatRoleType.AI:
                messages.append(Message(role=Role.ASSISTANT, content=chat.content))
        return messages

    async def process_chat_with_ai(
        self, session_id: str, user_question: str, context: str, user_id: str, setting: SettingModel = None
    ) -> Result[ChatHistoryModelDTO]:
        try:
            # Build prompt
            custom_system_prompt = getattr(setting, 'system_prompt', None)
            context_prompt = getattr(setting, 'context_prompt', None)
            final_system_prompt = self.prompt_builder.build_complete_system_prompt(
                context=context,
                custom_system_prompt=custom_system_prompt,
                context_prompt=context_prompt
            )

            # Không chặn gọi LLM, luôn gọi để AI tự trả lời (kể cả khi không có context)

            # Lấy lịch sử chat
            session_id = "session_admin" if user_id == "admin" else session_id
            chat_histories_dicts = (
                await self.db.chat_histories.find({"session_id": session_id})
                .sort("created_at", -1)
                .limit(21)
                .to_list(length=21)
            )
            chat_histories_dicts.reverse()
            chat_histories = [ChatHistoryModel.from_dict(d) for d in chat_histories_dicts]
            if chat_histories and chat_histories[-1].role == ChatRoleType.USER:
                chat_histories = chat_histories[:-1]

            # Chuyển lịch sử chat
            history_messages = self._convert_chat_history_to_messages(chat_histories)

            # Cấu hình LLM
            llm_config = GeminiConfig(
                temperature=setting.temperature,
                max_tokens=setting.max_tokens,
            )
            llm_manager = GeminiChatManager(config=llm_config)
            llm_manager.set_system_prompt(user_id, final_system_prompt)

            # Gọi LLM
            try:
                response = await llm_manager.chat(
                    user_id=user_id,
                    user_message=user_question,
                    history=history_messages,
                )
                ai_content = response.content
            except Exception as e:
                self.logger.error(f"Lỗi khi gọi LLM: {e}")
                ai_content = "Xin lỗi, tôi gặp sự cố kỹ thuật. Vui lòng thử lại sau."

            # Lưu phản hồi
            chat_assistant = ChatHistoryModel(
                session_id=session_id,
                user_id=user_id,
                content=ai_content,
                role=ChatRoleType.AI,
            )
            await self.db.chat_histories.insert_one(chat_assistant.to_dict())
            chat_history_dto = ChatHistoryModelDTO.from_model(chat_assistant)

            return Result.success(
                message=ChatResult.CHAT_CREATED.message,
                code=ChatResult.CHAT_CREATED.code,
                data=chat_history_dto,
            )
        except Exception as e:
            self.logger.error(f"Lỗi khi xử lý cuộc trò chuyện với AI: {e}", exc_info=True)
            return Result.failure(
                message="Lỗi khi xử lý với AI", code="ai_processing_error"
            )
