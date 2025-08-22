"""
Delete Session Command Handler - Xử lý lệnh xóa phiên trò chuyện

File này định nghĩa handler để xử lý DeleteSessionCommand, thực hiện việc
xóa phiên trò chuyện.
"""

from bson import ObjectId
from core.cqrs.base import CommandHandler
from ..delete_session_command import DeleteSessionCommand
from core.cqrs import CommandRegistry
from core.result import Result
from shared.messages import SessionChatMessage
from utils import get_logger
from app.database import get_collections


@CommandRegistry.register_handler(DeleteSessionCommand)
class DeleteSessionCommandHandler(CommandHandler):
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

    async def execute(self, command: DeleteSessionCommand) -> Result[None]:
        try:
            if not ObjectId.is_valid(command.session_id):
                return Result.failure(
                    message="Phiên trò chuyện không hợp lệ",
                    code="error",
                )

            # Kiểm tra xem phiên trò chuyện đã tồn tại chưa
            is_session_exists = await self.db.chat_sessions.delete_one(
                {"_id": ObjectId(command.session_id)}
            )
            if is_session_exists == 0:
                return Result.failure(
                    message=SessionChatMessage.SESSION_DELETED_FAILED.message,
                    code=SessionChatMessage.SESSION_DELETED_FAILED.code,
                )

            await self.db.chat_histories.delete_many({"session_id": command.session_id})

            return Result.success(
                message=SessionChatMessage.SESSION_DELETED_SUCCESS.message,
                code=SessionChatMessage.SESSION_DELETED_SUCCESS.code,
            )

        except Exception as e:
            self.logger.error(f"Lỗi khi tạo phiên trò chuyện: {e}", exc_info=True)
            return Result.failure(message="Lỗi hệ thống", code="error")
