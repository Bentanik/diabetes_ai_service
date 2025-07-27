"""
Delete Knowledge Command Handler - Xử lý xóa cơ sở tri thức

File này định nghĩa DeleteKnowledgeCommandHandler. Handler này thực hiện logic xóa cơ sở tri thức khỏi database với validation và error handling.

Chức năng chính:
- Validate ObjectId format
- Thực hiện xóa cơ sở tri thức theo ID
- Kiểm tra kết quả xóa
- Trả về kết quả thành công hoặc lỗi
"""

from bson import ObjectId

from app.database.manager import get_collections
from app.feature.knowledge import DeleteKnowledgeCommand
from core.cqrs import CommandRegistry
from core.cqrs.base import CommandHandler
from core.result.result import Result
from shared.messages.knowledge_message import KnowledgeResult
from utils import get_logger


@CommandRegistry.register_handler(DeleteKnowledgeCommand)
class DeleteKnowledgeCommandHandler(CommandHandler):
    """
    Handler để xử lý DeleteKnowledgeCommand
    """
    
    def __init__(self):
        """Khởi tạo handler"""
        super().__init__()
        self.logger = get_logger(__name__)

    async def execute(self, command: DeleteKnowledgeCommand) -> Result[None]:
        """
        Thực hiện xóa cơ sở tri thức
        
        Method này thực hiện các bước sau:
        1. Validate ObjectId format
        2. Thực hiện xóa cơ sở tri thức theo ID
        3. Kiểm tra kết quả xóa
        4. Trả về kết quả thành công hoặc lỗi
        
        Args:
            command (DeleteKnowledgeCommand): Command chứa ID cơ sở tri thức cần xóa

        Returns:
            Result[None]: Kết quả thành công hoặc lỗi với message và code tương ứng
        """
        self.logger.info(f"Xóa cơ sở tri thức: {command.id}")

        # Validate ObjectId format
        if not ObjectId.is_valid(command.id):
            self.logger.warning(f"ID không hợp lệ: {command.id}")
            return Result.failure(
                message=KnowledgeResult.NOT_FOUND.message,
                code=KnowledgeResult.NOT_FOUND.code,
            )

        # Lấy collection để thao tác với database
        collection = get_collections()

        # Thực hiện xóa bản ghi theo _id
        delete_result = await collection.knowledges.delete_one(
            {"_id": ObjectId(command.id)}
        )

        # Kiểm tra kết quả xóa
        if delete_result.deleted_count == 0:
            self.logger.warning(f"Không tìm thấy cơ sở tri thức với id: {command.id}")
            return Result.failure(
                message=KnowledgeResult.NOT_FOUND.message,
                code=KnowledgeResult.NOT_FOUND.code,
            )

        self.logger.info(f"Cơ sở tri thức đã được xóa: {command.id}")

        # Trả về kết quả thành công
        return Result.success(
            message=KnowledgeResult.DELETED.message,
            code=KnowledgeResult.DELETED.code,
            data=None,
        )
