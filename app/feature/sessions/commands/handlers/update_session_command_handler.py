"""
Update Session Command Handler - Xử lý lệnh cập nhật phiên trò chuyện

File này định nghĩa handler để xử lý UpdateSessionCommand, thực hiện việc
cập nhật phiên trò chuyện.
"""

from bson import ObjectId
from app.database.models import ChatSessionModel
from core.cqrs.base import CommandHandler
from ..update_session_command import UpdateSessionCommand
from core.cqrs import CommandRegistry
from core.result import Result
from shared.messages import SessionChatMessage
from utils import get_logger
from app.database import get_collections


@CommandRegistry.register_handler(UpdateSessionCommand)
class UpdateSessionCommandHandler(CommandHandler):
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

    async def execute(self, command: UpdateSessionCommand) -> Result[None]:
        try:
            if not ObjectId.is_valid(command.session_id):
                return Result.failure(
                    message="Phiên trò chuyện không hợp lệ",
                    code="error",
                )

            # Kiểm tra xem phiên trò chuyện đã tồn tại chưa
            update_data = {}
            if command.title is not None:
                update_data["title"] = command.title
            if command.external_knowledge is not None:
                update_data["external_knowledge"] = command.external_knowledge

            if update_data:
                is_session_exists = await self.db.chat_sessions.find_one_and_update(
                    {"_id": ObjectId(command.session_id)},
                    {"$set": update_data},
                    return_document=True
                )
            else:
                # Nếu không có gì để update, chỉ tìm document
                is_session_exists = await self.db.chat_sessions.find_one(
                    {"_id": ObjectId(command.session_id)}
                )

            if is_session_exists is None:
                return Result.failure(
                    message=SessionChatMessage.SESSION_UPDATED_FAILED.message,
                    code=SessionChatMessage.SESSION_UPDATED_FAILED.code,
                )

            return Result.success(
                message=SessionChatMessage.SESSION_UPDATED_SUCCESS.message,
                code=SessionChatMessage.SESSION_UPDATED_SUCCESS.code,
            )

        except Exception as e:
            self.logger.error(f"Lỗi khi tạo phiên trò chuyện: {e}", exc_info=True)
            return Result.failure(message="Lỗi hệ thống", code="error")
