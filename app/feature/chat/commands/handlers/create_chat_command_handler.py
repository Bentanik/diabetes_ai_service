"""
Chat Command Handler - Xử lý lệnh trò chuyện

File này định nghĩa handler để xử lý ChatCommand, thực hiện việc
trò chuyện với AI sử dụng RAG (Retrieval-Augmented Generation).
"""

from typing import List
from bson import ObjectId
from app.database.enums import ChatRoleType
from app.database.models import ChatHistoryModel
from core.cqrs import CommandHandler
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
        self.system_message = """
Bạn là một chuyên gia y tế thân thiện và chuyên nghiệp chuyên về bệnh tiểu đường.

Nhiệm vụ của bạn:
- Trả lời bằng tiếng Việt một cách rõ ràng, dễ hiểu và thân thiện, như khi tư vấn cho bệnh nhân hoặc gia đình họ.
- Tránh sử dụng thuật ngữ y khoa khó hiểu. Nếu cần thiết, hãy giải thích các thuật ngữ bằng ngôn ngữ đơn giản và dễ nhớ.
- Chỉ trả lời các câu hỏi liên quan đến bệnh tiểu đường.
- Hãy cung cấp câu trả lời ngắn gọn, tập trung trong khoảng 150-250 từ. Tránh lan man hoặc giải thích dài dòng.

Quan trọng:
- Dựa trên kiến thức chuyên môn của bạn và thông tin tham khảo được cung cấp để đưa ra câu trả lời chính xác.
- Nếu câu hỏi nằm ngoài phạm vi về tiểu đường, hãy lịch sự giải thích rằng bạn chỉ có thể hỗ trợ các chủ đề liên quan đến tiểu đường.
- Thể hiện sự đồng cảm và đưa ra lời khuyên thiết thực khi thích hợp, bao gồm khuyến khích người dùng tham khảo ý kiến chuyên gia y tế để chẩn đoán hoặc điều trị.
- Khi có thông tin tham khảo, hãy ưu tiên sử dụng thông tin đó để đưa ra câu trả lời chính xác và cập nhật nhất.
""".strip()

    def build_system_prompt_with_context(self, context: str = None) -> str:
        """
        Xây dựng system prompt với context từ RAG
        """
        base_prompt = self.system_message

        # Nếu có context từ RAG, thêm vào system prompt
        if context and context.strip():
            context_section = f"""

Thông tin tham khảo từ cơ sở dữ liệu y tế:
{context.strip()}

Hãy sử dụng thông tin tham khảo trên để trả lời câu hỏi của người dùng một cách chính xác và đầy đủ nhất.
"""
            return base_prompt + context_section

        return base_prompt


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
        self.llm_manager = GeminiChatManager()
        self.prompt_builder = DiabetesRAGPrompt()

    async def execute(self, command: CreateChatCommand) -> Result[None]:
        try:
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

            # Lấy settings để cấu hình RAG
            setting = await self.db.settings.find_one({})
            if not setting:
                return Result.failure(
                    message=SettingResult.NOT_FOUND.message,
                    code=SettingResult.NOT_FOUND.code,
                    data=[],
                )

            # Thực hiện RAG search để lấy context
            try:
                retrieved_documents = await self.vector_operations.search(
                    query_text=command.content,
                    top_k=setting.get("number_of_passages", 5),
                    score_threshold=setting.get("search_accuracy", 50) / 100,
                )

                # Kết hợp context từ các documents
                context = self._combine_retrieved_context(retrieved_documents)
                self.logger.info(
                    f"Retrieved {len(retrieved_documents)} documents for context"
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
            # Giả sử document có structure: {"content": "...", "metadata": {...}}
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
        self, session_id: str, user_question: str, context: str, user_id: str
    ) -> Result[ChatHistoryModelDTO]:
        """
        Xử lý chat với AI sử dụng RAG context
        """
        try:
            # Lấy lịch sử 20 cuộc trò chuyện gần nhất từ database (không bao gồm câu hỏi vừa thêm)
            chat_histories_dicts = (
                await self.db.chat_histories.find({"session_id": session_id})
                .sort("created_at", -1)
                .limit(21)  # Lấy 21 để loại bỏ câu hỏi vừa thêm
                .to_list(length=21)
            )

            chat_histories_dicts.reverse()
            chat_histories = [
                ChatHistoryModel.from_dict(d) for d in chat_histories_dicts
            ]

            # Loại bỏ câu hỏi vừa thêm (câu cuối cùng)
            if chat_histories and chat_histories[-1].role == ChatRoleType.USER:
                chat_histories = chat_histories[:-1]

            # Tạo system prompt với context
            system_prompt = self.prompt_builder.build_system_prompt_with_context(
                context
            )

            # Set system prompt cho user
            self.llm_manager.set_system_prompt(user_id, system_prompt)

            if context:
                self.logger.info("Sử dụng RAG context cho câu trả lời")
            else:
                self.logger.info("Không có RAG context, sử dụng kiến thức cơ bản")

            # Chuyển đổi lịch sử chat thành format Message
            history_messages = self._convert_chat_history_to_messages(chat_histories)

            # Gọi LLM với lịch sử và câu hỏi mới
            try:
                response = await self.llm_manager.chat(
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
