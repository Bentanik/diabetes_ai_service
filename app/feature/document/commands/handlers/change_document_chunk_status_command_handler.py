from bson import ObjectId
from app.database.manager import get_collections
from core.cqrs import CommandHandler, CommandRegistry
from core.result import Result
from rag.vector_store import VectorStoreManager
from utils import get_logger
from ..change_document_status_command import ChangeDocumentChunkStatusCommand


@CommandRegistry.register_handler(ChangeDocumentChunkStatusCommand)
class ChangeDocumentChunkStatusCommandHandler(CommandHandler):
    def __init__(self):
        super().__init__()
        self.db = get_collections()
        self.vector_store_manager = VectorStoreManager()
        self.logger = get_logger(__name__)

    async def execute(self, command: ChangeDocumentChunkStatusCommand) -> Result[None]:
        if not command.document_chunk_ids:
            return Result.failure("Danh sách document_chunk_ids không được rỗng")

        try:
            chunk_ids = [ObjectId(cid) for cid in command.document_chunk_ids]

            # B1: Cập nhật trạng thái trong MongoDB
            result = await self.db.document_chunks.update_many(
                {"_id": {"$in": chunk_ids}},
                [{"$set": {"is_active": {"$not": "$is_active"}}}]
            )

            if result.matched_count == 0:
                return Result.failure("Không tìm thấy chunk nào để cập nhật")

            # B2: Lấy lại dữ liệu đã cập nhật từ DB
            chunks = await self.db.document_chunks.find({"_id": {"$in": chunk_ids}}).to_list(None)

            if not chunks:
                return Result.failure("Không thể lấy lại dữ liệu sau khi cập nhật")

            # B3: Đồng bộ lên Qdrant
            for chunk in chunks:
                await self.vector_store_manager.update_payload_async(
                    collection_name=str(chunk["knowledge_id"]),
                    payload_updates={
                        "metadata": {
                            "document_chunk_id": str(chunk["_id"]),
                            "document_id": chunk["document_id"],
                            "knowledge_id": str(chunk["knowledge_id"]),
                            "is_active": chunk["is_active"]
                        }
                    },
                    metadata__document_chunk_id=str(chunk["_id"])
                )

        except Exception as e:
            self.logger.error(f"Lỗi khi thay đổi trạng thái tài liệu: {e}", exc_info=True)
            return Result.failure(f"Lỗi xử lý: {str(e)}", "error_processing")

        return Result.success(None)