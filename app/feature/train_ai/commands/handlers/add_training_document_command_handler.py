from bson import ObjectId
from rag.chunking import Chunk, get_chunking_instance
from app.database import get_collections
from app.database.models import DocumentParserModel
from core.cqrs import CommandHandler, CommandRegistry
from core.result import Result
from rag.vector_store import VectorStoreOperations
from ..add_training_document_command import AddTrainingDocumentCommand
from shared.messages import DocumentResult
from utils import get_logger


@CommandRegistry.register_handler(AddTrainingDocumentCommand)
class AddTrainingDocumentCommandHandler(CommandHandler):
    def __init__(self):
        super().__init__()
        self.logger = get_logger(__name__)
        self.vector_operations = VectorStoreOperations()
        self.collections = get_collections()

    async def execute(self, command: AddTrainingDocumentCommand) -> Result[None]:
        self.logger.info(f"Thêm tài liệu vào vector database: {command.document_id}")

        try:
            # 1. Lấy document từ MongoDB
            document_parsers = await self.collections.document_parsers.find(
                {"document_id": command.document_id}
            ).to_list(length=None)

            document = await self.collections.documents.find_one(
                {"_id": ObjectId(command.document_id)},
                {"knowledge_id": 1},
            )

            if not document:
                self.logger.info(
                    f"Không tìm thấy document với ID: {command.document_id}"
                )
                return Result.failure(
                    message=DocumentResult.NOT_FOUND.message,
                    code=DocumentResult.NOT_FOUND.code,
                )

            if not document_parsers:
                self.logger.info(
                    f"Không tìm thấy document parser với Document ID: {command.document_id}"
                )
                return Result.failure(
                    message=DocumentResult.NOT_FOUND.message,
                    code=DocumentResult.NOT_FOUND.code,
                )

            document_parsers = [
                DocumentParserModel.from_dict(doc) for doc in document_parsers
            ]

            # 2. Chunk tài liệu
            chunking = await get_chunking_instance()
            all_chunks = []

            for parser in document_parsers:
                chunks = await chunking.chunk_text(
                    text=parser.content,
                    metadata={
                        "document_parser_id": str(parser.id),
                        "is_active": parser.is_active,
                    },
                )
                all_chunks.extend(chunks)

            if not all_chunks:
                self.logger.info(
                    f"Không tạo được chunk nào từ tài liệu {command.document_id}"
                )
                return Result.failure(
                    message=DocumentResult.TRAINING_FAILED.message,
                    code=DocumentResult.TRAINING_FAILED.code,
                )

            # 3. Lưu vào vector database (Qdrant)
            texts = [chunk["text"] for chunk in all_chunks]
            metadatas = [chunk.get("metadata", {}) for chunk in all_chunks]
            print(document["knowledge_id"])
            collection_name = document["knowledge_id"]

            self.vector_operations.store_vectors(
                texts=texts,
                collection_name=collection_name,
                metadatas=metadatas,
            )

            # 4. Thành công
            self.logger.info(
                f"Đã lưu {len(texts)} vector vào collection {collection_name}"
            )
            return Result.success(
                message=DocumentResult.TRAINING_COMPLETED.message,
                code=DocumentResult.TRAINING_COMPLETED.code,
            )

        except Exception as e:
            self.logger.error(f"Lỗi xử lý tài liệu huấn luyện: {e}", exc_info=True)
            return Result.failure(
                message=DocumentResult.TRAINING_FAILED.message,
                code=DocumentResult.TRAINING_FAILED.code,
            )
