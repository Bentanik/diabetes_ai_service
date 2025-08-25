"""
Delete Document Command Handler - Xử lý command xóa tài liệu

File này định nghĩa handler để xử lý DeleteDocumentCommand, thực hiện việc
xóa một tài liệu khỏi database và storage.
"""

from datetime import datetime
from bson import ObjectId
from app.database import get_collections
from app.storage import MinioManager
from ..delete_document_command import DeleteDocumentCommand
from core.cqrs import CommandHandler, CommandRegistry
from core.result import Result
from shared.messages import DocumentMessage
from utils import get_logger
from rag.vector_store import VectorStoreManager


@CommandRegistry.register_handler(DeleteDocumentCommand)
class DeleteDocumentCommandHandler(CommandHandler):
    """
    Handler xử lý DeleteDocumentCommand để xóa tài liệu.
    """

    def __init__(self):
        """
        Khởi tạo handler
        """
        super().__init__()
        self.logger = get_logger(__name__)
        self.vector_store_manager = VectorStoreManager()

    async def execute(self, command: DeleteDocumentCommand) -> Result[None]:
        """
        Thực thi command xóa tài liệu

        Args:
            command (DeleteDocumentCommand): Command chứa ID tài liệu cần xóa

        Returns:
            Result[None]: Kết quả thành công hoặc lỗi
        """
        try:
            self.logger.info(f"Xóa tài liệu: {command.id}")

            # Kiểm tra tính hợp lệ của ID
            if not ObjectId.is_valid(command.id):
                return Result.failure(message="ID không hợp lệ", code="invalid_id")

            # Truy vấn database
            collections = get_collections()

            # Lấy thông tin tài liệu trước khi xóa
            document = await collections.documents.find_one(
                {"_id": ObjectId(command.id)}
            )
            if not document:
                return Result.failure(
                    message=DocumentMessage.NOT_FOUND.message,
                    code=DocumentMessage.NOT_FOUND.code,
                )

            # Xóa file từ storage nếu có
            if document.get("file_path"):
                try:
                    await self._delete_file_from_storage(document["file_path"])
                except Exception as e:
                    self.logger.warning(f"Không thể xóa file từ storage: {e}")
                    # Vẫn tiếp tục xóa record trong database

            # Xóa tài liệu từ database
            result = await collections.documents.delete_one(
                {"_id": ObjectId(command.id)}
            )

            # Xóa tài liệu từ vector store
            await self._delete_document_from_vector_store(
                document_id=str(document["_id"]), knowledge_id=document["knowledge_id"])

            # Xóa tài liệu làm sạch
            await collections.document_chunks.delete_many({"document_id": command.id})

            # Chuyển bên job document sang is delete
            await collections.document_jobs.update_one(
                {"document_id": command.id},
                {"$set": {"document_status": "deleted"}},
            )

            if result.deleted_count > 0:
                # Cập nhật thống kê của knowledge
                await self._update_knowledge_stats(
                    document["knowledge_id"], collections
                )

                return Result.success(
                    message=DocumentMessage.DELETED.message,
                    code=DocumentMessage.DELETED.code,
                )
            else:
                return Result.failure(
                    message="Không thể xóa tài liệu", code="delete_failed"
                )

        except Exception as e:
            self.logger.error(f"Lỗi khi xóa tài liệu: {e}", exc_info=True)
            return Result.failure(message="Lỗi hệ thống", code="error")

    async def _delete_file_from_storage(self, file_path: str):
        """
        Xóa file từ MinIO storage

        Args:
            file_path (str): Đường dẫn file trong storage (format: bucket_name/object_name)
        """
        try:
            # Parse file path: bucket_name/object_name
            path_parts = file_path.split("/", 1)
            if len(path_parts) == 2:
                bucket_name, object_name = path_parts

                # Xóa file từ MinIO
                minio_client = MinioManager.get_instance()
                minio_client.delete_file(bucket_name, object_name)

                self.logger.info(f"Đã xóa file từ storage: {file_path}")

            else:
                self.logger.warning(f"Định dạng file path không hợp lệ: {file_path}")

        except Exception as e:
            self.logger.error(f"Lỗi khi xóa file từ storage: {e}")
            raise

    async def _delete_document_from_vector_store(self, document_id: str, knowledge_id: str):
        """
        Xóa document từ vector store
        """
        try:
            await self.vector_store_manager.delete_by_metadata_async(
                collection_name=knowledge_id,
                metadata__document_id=document_id
            )
            self.logger.info(f"Đã xóa document từ vector store: {document_id}")
        except Exception as e:
            self.logger.error(f"Lỗi khi xóa document từ vector store: {e}")
            raise

    async def _update_knowledge_stats(self, knowledge_id: str, collections):
        """
        Cập nhật thống kê của knowledge sau khi xóa document

        Args:
            knowledge_id (str): ID của knowledge
            collections: Database collections
        """
        try:
            # Đếm số lượng documents còn lại
            doc_count = await collections.documents.count_documents(
                {"knowledge_id": knowledge_id}
            )

            # Tính tổng size của các documents còn lại
            pipeline = [
                {"$match": {"knowledge_id": knowledge_id}},
                {"$group": {"_id": None, "total_size": {"$sum": "$file_size_bytes"}}},
            ]
            result = await collections.documents.aggregate(pipeline).to_list(1)
            total_size = result[0]["total_size"] if result else 0

            # Cập nhật knowledge stats
            await collections.knowledges.update_one(
                {"_id": ObjectId(knowledge_id)},
                {
                    "$set": {
                        "document_count": doc_count,
                        "total_size_bytes": total_size,
                        "updated_at": datetime.now(),
                    }
                },
            )

            self.logger.info(
                f"Đã cập nhật stats cho knowledge {knowledge_id}: {doc_count} docs, {total_size} bytes"
            )

        except Exception as e:
            self.logger.error(f"Lỗi khi cập nhật knowledge stats: {e}")
