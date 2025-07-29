"""
Update Document Command Handler - Xử lý command cập nhật tài liệu

File này định nghĩa handler để xử lý UpdateDocumentCommand, thực hiện việc
cập nhật thông tin của một tài liệu trong database.
"""

from bson import ObjectId
from app.database import get_collections
from app.database.models import DocumentModel
from ..update_document_command import UpdateDocumentCommand
from core.cqrs import CommandHandler, CommandRegistry
from core.result import Result
from shared.messages import DocumentResult
from utils import get_logger


@CommandRegistry.register_handler(UpdateDocumentCommand)
class UpdateDocumentCommandHandler(CommandHandler):
    """
    Handler xử lý UpdateDocumentCommand để cập nhật thông tin tài liệu.
    """

    def __init__(self):
        """
        Khởi tạo handler
        """
        super().__init__()
        self.logger = get_logger(__name__)

    async def execute(self, command: UpdateDocumentCommand) -> Result[None]:
        """
        Thực thi command cập nhật tài liệu

        Args:
            command (UpdateDocumentCommand): Command chứa thông tin cập nhật

        Returns:
            Result[None]: Kết quả thành công hoặc lỗi
        """
        try:
            self.logger.info(f"Cập nhật tài liệu: {command.id}")

            # Kiểm tra tính hợp lệ của ID
            if not ObjectId.is_valid(command.id):
                return Result.failure(message="ID không hợp lệ", code="invalid_id")

            # Truy vấn database
            collections = get_collections()

            # Kiểm tra tài liệu có tồn tại không
            existing_doc = await collections.documents.find_one(
                {"_id": ObjectId(command.id)}
            )
            if not existing_doc:
                return Result.failure(
                    message=DocumentResult.NOT_FOUND.message,
                    code=DocumentResult.NOT_FOUND.code,
                )

            # Xây dựng update data
            update_data = {}
            if command.title is not None:
                update_data["title"] = command.title
            if command.description is not None:
                update_data["description"] = command.description
            if command.priority_diabetes is not None:
                update_data["priority_diabetes"] = command.priority_diabetes

            # Nếu không có gì để update
            if not update_data:
                return Result.success(
                    message="Không có thay đổi nào", code="no_changes"
                )

            # Thêm updated_at timestamp
            from datetime import datetime

            update_data["updated_at"] = datetime.now()

            # Thực hiện update
            result = await collections.documents.update_one(
                {"_id": ObjectId(command.id)}, {"$set": update_data}
            )

            if result.modified_count > 0:
                return Result.success(
                    message=DocumentResult.UPDATED.message,
                    code=DocumentResult.UPDATED.code,
                )
            else:
                return Result.failure(
                    message="Không thể cập nhật tài liệu", code="update_failed"
                )

        except Exception as e:
            self.logger.error(f"Lỗi khi cập nhật tài liệu: {e}", exc_info=True)
            return Result.failure(message="Lỗi hệ thống", code="error")
