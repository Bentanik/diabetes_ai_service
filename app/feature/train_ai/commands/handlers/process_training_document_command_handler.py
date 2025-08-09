from typing import List
from bson import ObjectId
from app.database.models import DocumentJobModel
from app.database.enums import DocumentJobStatus
from app.database import get_collections
from app.database.models import DocumentParserModel
from app.dto.enums.document_type import DocumentType
from core.cqrs import CommandHandler, CommandRegistry
from core.embedding.embedding_model import EmbeddingModel
from core.result import Result
from rag.chunking import Chunking
from rag.config.chunking_config import ChunkingConfig
from rag.schemas.pdf.text_block import TextBlock
from rag.vector_store import VectorStoreOperations
from ..process_training_document_command import ProcessTrainingDocumentCommand
from shared.messages import DocumentResult
from utils import get_logger
from rag.embedding import get_embedding_instance


@CommandRegistry.register_handler(ProcessTrainingDocumentCommand)
class ProcessTrainingDocumentCommandHandler(CommandHandler):
    def __init__(self):
        super().__init__()
        self.logger = get_logger(__name__)
        self.vector_operations = VectorStoreOperations.get_instance()
        self.collections = get_collections()

    async def execute(self, command: ProcessTrainingDocumentCommand) -> Result[None]:
        self.logger.info(
            f"Thêm tài liệu vào vector database: {command.document_job_id}"
        )

        try:
            if not ObjectId.is_valid(command.document_job_id):
                self.logger.info(
                    f"ID của job tài liệu không hợp lệ: {command.document_job_id}"
                )
                return Result.failure(
                    message=DocumentResult.NOT_FOUND.message,
                    code=DocumentResult.NOT_FOUND.code,
                )

            # 1. Lấy document job từ MongoDB
            document_job = await self.collections.document_jobs.find_one(
                {"_id": ObjectId(command.document_job_id)}
            )

            if not document_job:
                self.logger.info(
                    f"Không tìm thấy document job với ID: {command.document_job_id}"
                )
                return Result.failure(
                    message=DocumentResult.NOT_FOUND.message,
                    code=DocumentResult.NOT_FOUND.code,
                )

            document_job = DocumentJobModel.from_dict(document_job)

            await self._update_document_job(
                document_job_id=command.document_job_id,
                status=DocumentJobStatus.PROCESSING,
                progress=20,
                progress_message="Lấy dữ liệu đã làm sạch",
            )

            # 2. Lấy document từ MongoDB
            document_parsers: List[DocumentParserModel] = (
                await self.collections.document_parsers.find(
                    {"document_id": document_job.document_id}
                ).to_list(length=None)
            )

            document = await self.collections.documents.find_one(
                {"_id": ObjectId(document_job.document_id)},
                {"knowledge_id": 1},
            )

            if not document:
                self.logger.info(
                    f"Không tìm thấy document với ID: {document_job.document_id}"
                )
                return Result.failure(
                    message=DocumentResult.NOT_FOUND.message,
                    code=DocumentResult.NOT_FOUND.code,
                )

            if not document_parsers:
                self.logger.info(
                    f"Không tìm thấy document parser với Document ID: {document_job.document_id}"
                )
                return Result.failure(
                    message=DocumentResult.NOT_FOUND.message,
                    code=DocumentResult.NOT_FOUND.code,
                )

            await self._update_document_job(
                document_job_id=command.document_job_id,
                status=DocumentJobStatus.PROCESSING,
                progress=50,
                progress_message="Xử lý dữ liệu đã làm sạch",
            )

            # 3. Chunk tài liệu
            chunking_config = ChunkingConfig(
                max_chunk_size=512, min_chunk_size=64, chunk_overlap=200
            )
            embedding_model = await EmbeddingModel.get_instance()
            chunking = Chunking(
                config=chunking_config, model_name=embedding_model.model_name
            )
            text_blocks = [
                TextBlock(
                    context=parser["content"],
                    block_id=str(parser["_id"]),
                )
                for parser in document_parsers
            ]

            chunks = await chunking.chunk_text(text_blocks)

            if not chunks:
                self.logger.info(
                    f"Không tạo được chunk nào từ tài liệu {command.document_job_id}"
                )
                return Result.failure(
                    message=DocumentResult.TRAINING_FAILED.message,
                    code=DocumentResult.TRAINING_FAILED.code,
                )

            await self._update_document_job(
                document_job_id=command.document_job_id,
                status=DocumentJobStatus.PROCESSING,
                progress=70,
                progress_message="Hệ thống đang ghi nhớ nội dung tài liệu...",
            )

            texts = [chunk.text for chunk in chunks]
            metadatas = [
                {
                    "is_active": True,
                    "document_id": document_job.document_id,
                    "document_parser_id": chunk.block_id,
                }
                for chunk in chunks
            ]
            collection_name = document["knowledge_id"]

            await self.vector_operations.store_vectors(
                texts=texts,
                collection_name=collection_name,
                metadatas=metadatas,
            )

            await self.collections.documents.update_one(
                {"_id": ObjectId(document_job.document_id)},
                {"$set": {"type": DocumentType.TRAINING}},
            )

            await self._update_document_job(
                document_job_id=command.document_job_id,
                status=DocumentJobStatus.COMPLETED,
                progress=100,
                progress_message="Hệ thống đã ghi nhớ nội dung tài liệu xong",
            )

            # 5. Thành công
            self.logger.info(
                f"Đã lưu {len(texts)} vector vào collection {collection_name}"
            )
            return Result.success(
                message=DocumentResult.TRAINING_COMPLETED.message,
                code=DocumentResult.TRAINING_COMPLETED.code,
            )

        except Exception as e:
            self.logger.error(f"Lỗi xử lý tài liệu huấn luyện: {e}", exc_info=True)
            await self._update_document_job(
                document_job_id=command.document_job_id,
                status=DocumentJobStatus.FAILED,
                progress_message=f"Lỗi: {str(e)}",
            )
            return Result.failure(
                message=DocumentResult.TRAINING_FAILED.message,
                code=DocumentResult.TRAINING_FAILED.code,
            )

    async def _update_document_job(
        self,
        document_job_id: str,
        status: DocumentJobStatus = None,
        progress: float = None,
        progress_message: str = None,
    ) -> None:

        # Update nested status fields giống upload handler
        set_fields = {}
        if status is not None:
            set_fields["status.status"] = status
        if progress is not None:
            set_fields["status.progress"] = progress
        if progress_message is not None:
            set_fields["status.progress_message"] = progress_message

        if set_fields:
            await self.collections.document_jobs.update_one(
                {"_id": ObjectId(document_job_id)},
                {"$set": set_fields},
            )
