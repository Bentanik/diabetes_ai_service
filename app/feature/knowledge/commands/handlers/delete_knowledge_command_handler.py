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
from rag.vector_store import VectorStoreOperations
from shared.messages.knowledge_message import KnowledgeResult
from utils import get_logger
from app.storage import MinioManager


@CommandRegistry.register_handler(DeleteKnowledgeCommand)
class DeleteKnowledgeCommandHandler(CommandHandler):
    """
    Handler để xử lý DeleteKnowledgeCommand
    """

    def __init__(self):
        """Khởi tạo handler"""
        super().__init__()
        self.vector_operations = VectorStoreOperations.get_instance()
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
                message=KnowledgeResult.NOT_FOUND.message,
                code=KnowledgeResult.NOT_FOUND.code,
            )

        # Thực hiện xóa
        await self.delete_knowledge(command)

        # Trả về kết quả thành công
        return Result.success(
            message=KnowledgeResult.DELETED.message,
            code=KnowledgeResult.DELETED.code,
            data=None,
        )


    async def delete_knowledge(self, command: DeleteKnowledgeCommand) -> Result[None]:
        """
        Thực hiện xóa cơ sở tri thức

        Args:
            command (DeleteKnowledgeCommand): Command chứa ID cơ sở tri thức cần xóa

        Returns:
            Result[None]: Kết quả thành công hoặc lỗi với message và code tương ứng
        """
        # Thực hiện xóa tài liệu trong MinIO
        documents = await self.collection.knowledges.find({"knowledge_id": ObjectId(command.id)}).to_list(length=None)
        
        if documents:
            for doc in documents:
                if "file_path" in doc and doc["file_path"]:
                    try:
                        self.minio_manager.delete_file(doc["file_path"])
                        self.logger.info(f"Đã xóa tài liệu: {doc['file_path']}")
                    except Exception as e:
                        self.logger.error(f"Lỗi khi xóa tài liệu {doc['file_path']}: {str(e)}")

        # Thực hiện xóa trong vector store
        self.vector_operations.delete_collection(command.id)

        # Thực hiện xóa document parser
        await self.collection.documents.delete_many({"knowledge_id": command.id})

        await self.collection.document_parsers.delete_many({"knowledge_id": command.id})

        # Thực hiện xóa bản ghi theo _id
        delete_result = await self.collection.knowledges.delete_one(
            {"_id": ObjectId(command.id)}
        )

        # Thực hiện xóa mềm trong document_jobs
        await self.collection.document_jobs.update_many(
            {"knowledge_id": command.id},
            {"$set": {"is_document_delete": True}}
        )

        # Kiểm tra kết quả xóa
        if delete_result.deleted_count == 0:
            self.logger.warning(f"Không tìm thấy cơ sở tri thức với id: {command.id}")
            return Result.failure(
                message=KnowledgeResult.NOT_FOUND.message,
                code=KnowledgeResult.NOT_FOUND.code,
            )

        # Xóa collection từ VectorStore
        self.vector_operations.delete_collection(command.id)

        self.logger.info(f"Cơ sở tri thức đã được xóa: {command.id}")

        return Result.success(
            message=KnowledgeResult.DELETED.message,
            code=KnowledgeResult.DELETED.code,
            data=None,
        )