"""
Create Session Command Handler - Xử lý lệnh tạo phiên trò chuyện

File này định nghĩa handler để xử lý CreateSessionCommand, thực hiện việc
tạo phiên trò chuyện mới.
"""

from app.database.models import ChatSessionModel
from core.cqrs.base import CommandHandler
from ..create_session_command import CreateSessionCommand
from core.cqrs import CommandRegistry
from core.result import Result
from shared.messages import SessionChatMessage
from utils import get_logger
from app.database import get_collections


@CommandRegistry.register_handler(CreateSessionCommand)
class CreateSessionCommandHandler(CommandHandler):
    """
    Handler xử lý lệnh tạo phiên trò chuyện.
    """

    def __init__(self):
        """
        Khởi tạo handler
        """
        super().__init__()
        self.db = get_collections()
        self.logger = get_logger(__name__)

    async def execute(self, command: CreateSessionCommand) -> Result[None]:
        try:
            # Kiểm tra xem phiên trò chuyện đã tồn tại chưa
            is_session_exists = await self.db.chat_sessions.count_documents(
                {"user_id": command.user_id, "title": command.title}
            )

            if is_session_exists > 0:
                return Result.failure(
                    message=SessionChatMessage.SESSION_ALREADY_EXISTS.message,
                    code=SessionChatMessage.SESSION_ALREADY_EXISTS.code,
                )

            # Tạo phiên trò chuyện mới
            session = ChatSessionModel(
                user_id=command.user_id,
                title=command.title,
            )

            # Lưu phiên trò chuyện vào database
            await self.db.chat_sessions.insert_one(session.to_dict())

            return Result.success(
                message=SessionChatMessage.SESSION_CREATED.message,
                code=SessionChatMessage.SESSION_CREATED.code,
            )

        except Exception as e:
            self.logger.error(f"Lỗi khi tạo phiên trò chuyện: {e}", exc_info=True)
            return Result.failure(message="Lỗi hệ thống", code="error")
