"""
Chat Command Handler - Xử lý lệnh trò chuyện

File này định nghĩa handler để xử lý ChatCommand, thực hiện việc
trò chuyện với AI.
"""

from typing import List
from bson import ObjectId
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from app.database.enums import ChatRoleType
from app.database.models import ChatHistoryModel
from core.cqrs import CommandHandler
from shared.messages import SessionChatResult
from ..create_chat_command import CreateChatCommand
from core.cqrs import CommandRegistry
from core.result import Result
from shared.messages import ChatResult
from utils import get_logger
from app.database import get_collections
from app.database.models import ChatSessionModel
from rag.retriever import Retriever
from core.llm.load_llm import get_gemini_llm


class DiabetesPrompt:
    def __init__(self):
        self.system_message = """
You are a friendly and professional medical expert specializing in diabetes.

Your task:
- Respond in Vietnamese clearly, understandably, and kindly, as if consulting with patients or their families.
- Avoid using difficult medical terminology. If necessary, explain terms in simple and memorable language.
- Only answer questions related to diabetes.
- Please provide concise, focused answers within 150-250 words. Avoid rambling or lengthy explanations.

Important:
- Base your answers on your expert knowledge. Do not fabricate or guess beyond what you know to be true.
- If the question is outside the scope of diabetes, politely explain that you can only assist with diabetes-related topics.
- Show empathy and offer practical advice where appropriate, including encouraging users to consult healthcare professionals for diagnosis or treatment.
""".strip()

    def build_messages(self, user_question: str) -> List:
        return [
            SystemMessage(content=self.system_message),
            HumanMessage(content=user_question.strip()),
        ]


@CommandRegistry.register_handler(CreateChatCommand)
class CreateChatCommandHandler(CommandHandler):
    """
    Handler xử lý trò chuyện với AI.
    """

    def __init__(self):
        """
        Khởi tạo handler
        """
        super().__init__()
        self.db = get_collections()
        self.logger = get_logger(__name__)
        self.retriever = Retriever()
        self.llm = get_gemini_llm()

    async def execute(self, command: CreateChatCommand) -> Result[None]:
        try:
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

            # Lưu cuộc trò chuyện của user vào database
            chat_user = ChatHistoryModel(
                session_id=session_id,
                user_id=command.user_id,
                content=command.content,
                role=ChatRoleType.USER,
            )
            await self.db.chat_histories.insert_one(chat_user.to_dict())

            await self.process_chat_with_ai(
                session_id=session_id,
                content=command.content,
                user_id=command.user_id,
            )

            return Result.success(
                message=ChatResult.CHAT_CREATED.message,
                code=ChatResult.CHAT_CREATED.code,
                data=None,
            )

        except Exception as e:
            self.logger.error(f"Lỗi khi tạo cuộc trò chuyện: {e}", exc_info=True)
            return Result.failure(message="Lỗi hệ thống", code="error")

    async def _create_session(self, command: CreateChatCommand) -> str:
        try:
            if not command.session_id:
                session = ChatSessionModel(
                    user_id=command.user_id,
                    title=command.content,
                )
                await self.db.chat_sessions.insert_one(session.to_dict())
                return session.id

            return command.session_id

        except Exception as e:
            self.logger.error(f"Lỗi khi tạo phiên trò chuyện: {e}", exc_info=True)
            return Result.failure(message="Lỗi hệ thống", code="error")

    async def process_chat_with_ai(
        self, session_id: str, content: str, user_id: str
    ) -> Result[None]:
        try:
            # Lấy lịch sử 20 cuộc trò chuyện gần nhất từ database
            chat_histories_dicts = (
                await self.db.chat_histories.find({"session_id": session_id})
                .sort("created_at", -1)
                .limit(20)
                .to_list(length=20)
            )

            chat_histories_dicts.reverse()

            chat_histories = [
                ChatHistoryModel.from_dict(d) for d in chat_histories_dicts
            ]

            # Chuẩn bị list messages, bắt đầu bằng system message
            messages = [SystemMessage(content=DiabetesPrompt().system_message)]

            for chat in chat_histories:
                if chat.role == ChatRoleType.USER:
                    messages.append(HumanMessage(content=chat.content))
                else:
                    messages.append(AIMessage(content=chat.content))

            messages.append(HumanMessage(content=content.strip()))

            # Gọi LLM
            response = await self.llm.ainvoke(messages)

            # Lưu response vào database
            chat_assistant = ChatHistoryModel(
                session_id=session_id,
                user_id=user_id,
                content=response.content,
                role=ChatRoleType.AI,
            )
            await self.db.chat_histories.insert_one(chat_assistant.to_dict())
            return Result.success(
                message=ChatResult.CHAT_CREATED.message,
                code=ChatResult.CHAT_CREATED.code,
                data=None,
            )

        except Exception as e:
            self.logger.error(f"Lỗi khi xử lý cuộc trò chuyện: {e}", exc_info=True)
            return Result.failure(message="Lỗi hệ thống", code="error")
