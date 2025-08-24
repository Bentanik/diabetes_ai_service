"""
Delete Knowledge Command Handler - Xử lý xóa cơ sở tri thức

File này định nghĩa DeleteKnowledgeCommandHandler. Handler này thực hiện logic xóa cơ sở tri thức khỏi database với validation và error handling.

Chức năng chính:
- Validate ObjectId format
- Thực hiện xóa cơ sở tri thức theo ID
- Kiểm tra kết quả xóa
- Trả về kết quả thành công hoặc lỗi
"""

import asyncio
from bson import ObjectId

from app.database.enums import DocumentStatus
from app.database.manager import get_collections
from app.feature.knowledge.commands import DeleteKnowledgeCommand
from core.cqrs import CommandRegistry, CommandHandler
from core.result import Result
from rag.vector_store import VectorStoreManager
from shared.messages.knowledge_message import KnowledgeMessage
from utils import get_logger
from app.storage import MinioManager
from app.database.models import DocumentJobModel

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

        # Xóa ids của cơ sở tri thức khỏi setting
        await self.collection.settings.update_one(
            {},
            {"$pull": {"list_knowledge_ids": command.id}}
        )

        # Xóa tài liệu trong cơ sở tri thức
        await self.delete_document(command)

        # Xóa collection từ VectorStore
        await self.vector_store_manager.delete_collection_async(name=command.id)

        self.logger.info(f"Cơ sở tri thức đã được xóa: {command.id}")

        return Result.success(
            message=KnowledgeMessage.DELETED.message,
            code=KnowledgeMessage.DELETED.code,
            data=None,
        )

    async def delete_document(self, command: DeleteKnowledgeCommand) -> bool:
        try:
            # Xóa dữ liệu trong Mongo song song
            await asyncio.gather(
                self.collection.documents.delete_many({"knowledge_id": command.id}),
                self.collection.document_chunks.delete_many({"knowledge_id": command.id}),
                self.collection.document_jobs.update_many(
                    {"knowledge_id": command.id},
                    {"$set": {"document_status": DocumentStatus.DELETED}}
                )
            )

            # Lấy danh sách document jobs
            document_jobs_cursor = self.collection.document_jobs.find({"knowledge_id": command.id})
            document_jobs = [
                DocumentJobModel.from_dict(doc)
                async for doc in document_jobs_cursor
            ]

            # Xóa file trong MinIO
            for job in document_jobs:
                if job.file and job.file.path:
                    try:
                        bucket, obj = job.file.path.split("/", 1)
                        self.minio_manager.delete_file(bucket_name=bucket, object_name=obj)
                    except Exception as minio_err:
                        self.logger.warning(f"Lỗi khi xóa file {job.file.path} trong MinIO: {minio_err}")

            return True

        except Exception as e:
            self.logger.error(f"Lỗi xóa tài liệu: {e}", exc_info=True)
            return False