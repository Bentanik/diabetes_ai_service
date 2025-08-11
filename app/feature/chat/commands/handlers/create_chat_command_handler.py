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
- Chỉ trả lời các câu hỏi liên quan đến bệnh tiểu đường
- Thể hiện sự đồng cảm và đưa ra lời khuyên thiết thực
- Khuyến khích người dùng tham khảo ý kiến chuyên gia y tế khi cần thiết
- Dựa trên kiến thức chuyên môn và thông tin tham khảo được cung cấp
""".strip()

    def build_complete_system_prompt(
        self, 
        context: Optional[str] = None,
        custom_system_prompt: Optional[str] = None,
        context_prompt: Optional[str] = None
    ) -> str:
        """
        Xây dựng system prompt hoàn chỉnh bằng cách kết hợp:
        1. Base system config (cố định)
        2. Custom system prompt từ database
        3. Context prompt + RAG context từ database

        Args:
            context: Nội dung context từ RAG search
            custom_system_prompt: System prompt tùy chỉnh từ database
            context_prompt: Template cho context từ database
        
        Returns:
            str: System prompt hoàn chỉnh
        """
        prompt_parts = []
        
        # 1. Luôn luôn có base system config
        prompt_parts.append(self.base_system_config)
        
        # 2. Thêm custom system prompt từ database nếu có
        if custom_system_prompt and custom_system_prompt.strip():
            prompt_parts.append("\n" + custom_system_prompt.strip())
        
        # 3. Thêm context information nếu có
        if context and context.strip():
            if context_prompt and context_prompt.strip():
                # Sử dụng context_prompt từ database
                if "{context}" in context_prompt:
                    context_section = context_prompt.format(context=context.strip())
                else:
                    context_section = f"{context_prompt}\n{context.strip()}"
            else:
                # Không có context_prompt từ database, skip context
                context_section = None
                
            if context_section:
                prompt_parts.append("\n" + context_section)
        
        return "\n".join(prompt_parts)


@CommandRegistry.register_handler(CreateChatCommand)
class CreateChatCommandHandler(CommandHandler):
    """
    Handler xử lý trò chuyện với AI sử dụng RAG.
    """

    def __init__(self):
        """
        Khởi tạo handler
        """
        super().__init__()
        self.db = get_collections()
        self.logger = get_logger(__name__)
        self.vector_operations = VectorStoreOperations.get_instance()
        self.prompt_builder = DiabetesRAGPrompt()

    async def execute(self, command: CreateChatCommand) -> Result[None]:
        try:
            session_id = "session_admin"
            if command.user_id is not None and command.user_id == "admin":
                chat_user = ChatHistoryModel(
                    session_id=session_id,
                    user_id=command.user_id,
                    content=command.content,
                    role=ChatRoleType.USER,
                )
                await self.db.chat_histories.insert_one(chat_user.to_dict())
                
            else:
                # Validate session_id
                if command.session_id and not ObjectId.is_valid(command.session_id):
                    return Result.failure(
                        message="Phiên trò chuyện không hợp lệ",
                        code="error",
                    )

                # Check if session exists
                if command.session_id:
                    is_session_exists = await self.db.chat_sessions.count_documents(
                        {"_id": ObjectId(command.session_id)}
                    )
                    if not is_session_exists:
                        return Result.failure(
                            message=SessionChatResult.SESSION_NOT_FOUND.message,
                            code=SessionChatResult.SESSION_NOT_FOUND.code,
                        )

                # Create or get session
                session_id = await self._create_session(command)

                # Lưu câu hỏi của user vào database
                chat_user = ChatHistoryModel(
                    session_id=session_id,
                    user_id=command.user_id,
                    content=command.content,
                    role=ChatRoleType.USER,
                )
                await self.db.chat_histories.insert_one(chat_user.to_dict())

            # Lấy settings từ database
            setting = await self.db.settings.find_one({})
            if not setting:
                return Result.failure(
                    message=SettingResult.NOT_FOUND.message,
                    code=SettingResult.NOT_FOUND.code,
                    data=[],
                )
            
            setting = SettingModel.from_dict(setting)

            # Thực hiện RAG search để lấy context
            context = None
            try:
                retrieved_documents = await self.vector_operations.search(
                    query_text=command.content,
                    top_k=setting.top_k,
                    score_threshold=setting.search_accuracy,
                    collection_names=setting.list_knowledge_ids if setting.list_knowledge_ids else None,
                )

                # Kết hợp context từ các documents
                context = self._combine_retrieved_context(retrieved_documents)
                self.logger.info(
                    f"Retrieved {len(retrieved_documents)} documents for RAG context"
                )

            except Exception as e:
                self.logger.warning(f"Lỗi khi tìm kiếm RAG context: {e}")
                context = None

            # Xử lý chat với AI
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
        """
        Kết hợp context từ các documents được retrieve
        """
        if not documents:
            return ""

        context_parts = []
        for i, doc in enumerate(documents, 1):
            content = doc.get("content", "") if isinstance(doc, dict) else str(doc)
            if content.strip():
                context_parts.append(f"[Tài liệu {i}]: {content.strip()}")

        return "\n\n".join(context_parts)

    async def _create_session(self, command: CreateChatCommand) -> str:
        """
        Tạo hoặc lấy session ID
        """
        try:
            if not command.session_id:
                session = ChatSessionModel(
                    user_id=command.user_id,
                    title=(
                        command.content[:100] + "..."
                        if len(command.content) > 100
                        else command.content
                    ),
                )
                await self.db.chat_sessions.insert_one(session.to_dict())
                return str(session.id)

            return command.session_id

        except Exception as e:
            self.logger.error(f"Lỗi khi tạo phiên trò chuyện: {e}", exc_info=True)
            raise

    def _convert_chat_history_to_messages(
        self, chat_histories: List[ChatHistoryModel]
    ) -> List[Message]:
        """
        Chuyển đổi lịch sử chat từ database thành format Message của Gemini
        """
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
        """
        Xử lý chat với AI sử dụng prompts từ database + base config
        """
        try:
            # Lấy lịch sử cuộc trò chuyện
            session_id = user_id == "admin" and "session_admin" or session_id
            chat_histories_dicts = (
                await self.db.chat_histories.find({"session_id": session_id})
                .sort("created_at", -1)
                .limit(21)
                .to_list(length=21)
            )

            chat_histories_dicts.reverse()
            chat_histories = [
                ChatHistoryModel.from_dict(d) for d in chat_histories_dicts
            ]

            # Loại bỏ câu hỏi vừa thêm (câu cuối cùng)
            if chat_histories and chat_histories[-1].role == ChatRoleType.USER:
                chat_histories = chat_histories[:-1]

            # Lấy prompts từ settings
            custom_system_prompt = getattr(setting, 'system_prompt', None)
            context_prompt = getattr(setting, 'context_prompt', None)

            # Xây dựng system prompt hoàn chỉnh
            final_system_prompt = self.prompt_builder.build_complete_system_prompt(
                context=context,
                custom_system_prompt=custom_system_prompt,
                context_prompt=context_prompt
            )

            # Log thông tin về prompt composition
            self.logger.info("=== PROMPT COMPOSITION ===")
            self.logger.info("Base system config: LOADED")
            
            if custom_system_prompt:
                self.logger.info(f"Custom system prompt: LOADED ({len(custom_system_prompt)} chars)")
            else:
                self.logger.info("Custom system prompt: NOT PROVIDED")

            if context and context_prompt:
                self.logger.info(f"Context prompt + RAG context: LOADED ({len(context)} chars)")
            elif context:
                self.logger.info("RAG context available but no context_prompt provided - SKIPPED")
            else:
                self.logger.info("No RAG context available")

            self.logger.info(f"Final prompt length: {len(final_system_prompt)} characters")

            # Cấu hình và gọi LLM
            llm_config = GeminiConfig(
                temperature=setting.temperature,
                max_tokens=setting.max_tokens,
            )
            llm_manager = GeminiChatManager(config=llm_config)
            llm_manager.set_system_prompt(user_id, final_system_prompt)

            # Chuyển đổi lịch sử chat
            history_messages = self._convert_chat_history_to_messages(chat_histories)

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

            # Lưu response vào database
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
            self.logger.error(
                f"Lỗi khi xử lý cuộc trò chuyện với AI: {e}", exc_info=True
            )
            return Result.failure(
                message="Lỗi khi xử lý với AI", code="ai_processing_error"
            )