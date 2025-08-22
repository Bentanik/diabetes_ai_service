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
from app.feature.knowledge.commands import DeleteKnowledgeCommand
from core.cqrs import CommandRegistry, CommandHandler
from core.result import Result
from rag.vector_store import VectorStoreManager
from shared.messages.knowledge_message import KnowledgeMessage
from utils import get_logger
from app.storage import MinioManager
from core.cqrs import Mediator

@CommandRegistry.register_handler(DeleteKnowledgeCommand)
class DeleteKnowledgeCommandHandler(CommandHandler):
    """
    Handler để xử lý DeleteKnowledgeCommand
    """

    def __init__(self):
        """Khởi tạo handler"""
        super().__init__()
        self.vector_store_manager = VectorStoreManager()
        self.collection = get_collections()
        self.logger = get_logger(__name__)
        self.minio_manager = MinioManager.get_instance()

    async def execute(self, command: DeleteKnowledgeCommand) -> Result[None]:
        """
        Thực hiện xóa cơ sở tri thức

        Method này thực hiện các bước sau:
        1. Validate ObjectId format
        2. Thực hiện xóa cơ sở tri thức theo ID
        3. Kiểm tra kết quả xóa
        4. Xóa tài liệu trong MinIO
        5. Xóa collection từ VectorStore
        6. Xóa Document và Document Parser
        7. Trả về kết quả thành công hoặc lỗi

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
                message=KnowledgeMessage.NOT_FOUND.message,
                code=KnowledgeMessage.NOT_FOUND.code,
            )

        # Thực hiện xóa
        response = await self.delete_knowledge(command)

        return response


    async def delete_knowledge(self, command: DeleteKnowledgeCommand) -> Result[None]:
        """
        Thực hiện xóa cơ sở tri thức

        Args:
            command (DeleteKnowledgeCommand): Command chứa ID cơ sở tri thức cần xóa

        Returns:
            Result[None]: Kết quả thành công hoặc lỗi với message và code tương ứng
        """

        delete_result = await self.collection.knowledges.delete_one(
            {"_id": ObjectId(command.id)}
        )

        # Kiểm tra kết quả xóa
        if delete_result.deleted_count == 0:
            return Result.failure(code=KnowledgeMessage.NOT_FOUND.code, message=KnowledgeMessage.NOT_FOUND.message)

        # Xóa tài liệu trong cơ sở tri thức
        # await self.delete_document()

        # Xóa collection từ VectorStore
        await self.vector_store_manager.delete_collection_async(command.id)

        self.logger.info(f"Cơ sở tri thức đã được xóa: {command.id}")

        return Result.success(
            message=KnowledgeMessage.DELETED.message,
            code=KnowledgeMessage.DELETED.code,
            data=None,
        )

    # async def delete_document(self) -> bool:
    #     from app.feature.document.commands import DeleteDocumentCommand
    #     try:
    #         command = DeleteDocumentCommand(knowledge_id=command.id)
    #         await Mediator.send(command)
    #         return True
    #     except Exception as e:
    #         self.logger.error(f"Lỗi xóa tài liệu: {str(e)}", exc_info=True)
    #         return False