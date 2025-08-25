from bson import ObjectId
import torch
from app.database.enums import DocumentJobStatus, DocumentType, DocumentJobType
from app.database import get_collections
from app.database.models import DocumentJobModel, DocumentChunkModel
from core.cqrs import CommandHandler, CommandRegistry
from core.embedding import EmbeddingModel
from core.result import Result
from rag.vector_store import VectorStoreManager
from ..process_training_document_command import ProcessTrainingDocumentCommand
from shared.messages import DocumentMessage
from utils import get_logger


@CommandRegistry.register_handler(ProcessTrainingDocumentCommand)
class ProcessTrainingDocumentCommandHandler(CommandHandler):
    def __init__(self):
        super().__init__()
        self.logger = get_logger(__name__)
        self.vector_store_manager = VectorStoreManager()
        self.embedding_model = EmbeddingModel()
        self.db = get_collections()

    async def execute(self, command: ProcessTrainingDocumentCommand) -> Result[None]:
        self.logger.info(
            f"Thêm tài liệu vào vector database: {command.document_job_id}"
        )

        try:
            # 1. Lấy document job từ MongoDB
            if not ObjectId.is_valid(command.document_job_id):
                self.logger.info(
                    f"ID của job tài liệu không hợp lệ: {command.document_job_id}"
                )
                return Result.failure(
                    message=DocumentMessage.NOT_FOUND.message,
                    code=DocumentMessage.NOT_FOUND.code,
                )

            document_job = await self.db.document_jobs.find_one(
                {"_id": ObjectId(command.document_job_id)}
            )

            if not document_job:
                self.logger.info(
                    f"Không tìm thấy document job với ID: {command.document_job_id}"
                )
                return Result.failure(
                    message=DocumentMessage.NOT_FOUND.message,
                    code=DocumentMessage.NOT_FOUND.code,
                )

            document_job = DocumentJobModel.from_dict(document_job)

            # Cập nhật trạng thái bắt đầu xử lý
            await self._update_document_job(
                document_job_id=command.document_job_id,
                status=DocumentJobStatus.PROCESSING,
                progress=15,
                progress_message="Bắt đầu xử lý tài liệu..."
            )

            # 2. Lấy document chunk
            await self._update_document_job(
                document_job_id=command.document_job_id,
                progress=20,
                progress_message="Đang tải các chunk tài liệu..."
            )

            document_chunks = await self.db.document_chunks.find(
                {"document_id": document_job.document_id}
            ).to_list(length=None)

            document_chunks = [DocumentChunkModel.from_dict(chunk) for chunk in document_chunks]

            if not document_chunks:
                self.logger.warning(f"Không tìm thấy chunk nào cho document: {document_job.document_id}")
                await self._update_document_job(
                    document_job_id=command.document_job_id,
                    status=DocumentJobStatus.FAILED,
                    progress_message="Không tìm thấy chunk tài liệu"
                )
                return Result.failure(
                    message="Không tìm thấy chunk tài liệu",
                    code="CHUNKS_NOT_FOUND"
                )

            # 3. Tạo embedding cho document chunk
            await self._update_document_job(
                document_job_id=command.document_job_id,
                progress=30,
                progress_message=f"Đang tạo embedding cho {len(document_chunks)} chunks..."
            )

            embeddings = []
            total_chunks = len(document_chunks)
            
            for i, chunk in enumerate(document_chunks):
                try:
                    embedding = await self.embedding_model.embed(chunk.content)
                    embeddings.append(embedding)
                    
                    # Cập nhật tiến độ embedding
                    embedding_progress = 0.3 + (0.5 * (i + 1) / total_chunks)
                    await self._update_document_job(
                        document_job_id=command.document_job_id,
                        progress=embedding_progress,
                        progress_message=f"Đã tạo embedding {i + 1}/{total_chunks} chunks"
                    )
                    
                    # Cleanup every 5 chunks
                    if i % 5 == 0:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    # Retry
                    embedding = await self.embedding_model.embed(chunk.content)
                    embeddings.append(embedding)

            # Final cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 4. Tạo payload cho document chunk
            await self._update_document_job(
                document_job_id=command.document_job_id,
                progress=80,
                progress_message="Đang chuẩn bị dữ liệu cho vector store..."
            )

            payloads = [
                {
                    "content": chunk.content,
                    "metadata": {
                        "document_chunk_id": str(chunk.id),
                        "document_id": str(document_job.document_id),
                        "knowledge_id": str(document_job.knowledge_id),
                        "is_active": chunk.is_active,
                    },
                    "document_is_active": True,
                }
                for chunk in document_chunks
            ]

            # 5. Lưu xuống vector store
            await self._update_document_job(
                document_job_id=command.document_job_id,
                progress=90,
                progress_message="Đang lưu vào vector database..."
            )

            await self.vector_store_manager.insert_async(
                name=document_job.knowledge_id,
                embeddings=embeddings,
                payloads=payloads,
            )

            # 6. Thay đổi trạng thái của Document
            await self.db.documents.update_one(
                {"_id": ObjectId(document_job.document_id)},
                {"$set": {"document_type": DocumentType.TRAINED}},
            )

            # 7. Cập nhật trạng thái hoàn thành
            await self._update_document_job(
                document_job_id=command.document_job_id,
                status=DocumentJobStatus.COMPLETED,
                progress=100,
                progress_message="Hoàn thành xử lý tài liệu",
            )
            
            return Result.success(
                message=DocumentMessage.TRAINING_COMPLETED.message,
                code=DocumentMessage.TRAINING_COMPLETED.code,
            )

        except Exception as e:
            self.logger.error(f"Lỗi xử lý tài liệu huấn luyện: {e}", exc_info=True)
            await self._update_document_job(
                document_job_id=command.document_job_id,
                status=DocumentJobStatus.FAILED,
                progress_message=f"Lỗi: {str(e)}",
            )
            return Result.failure(
                message=DocumentMessage.TRAINING_FAILED.message,
                code=DocumentMessage.TRAINING_FAILED.code,
            )

    async def _update_document_job(
        self, 
        document_job_id: str, 
        status: DocumentJobStatus = None,
        progress: float = None,
        progress_message: str = None,
    ):
        """Cập nhật trạng thái của document job"""
        try:
            update_data = {}
            
            if status is not None:
                update_data["processing_status.status"] = status.value
            
            if progress is not None:
                update_data["processing_status.progress"] = progress
                
            if progress_message is not None:
                update_data["processing_status.progress_message"] = progress_message
            
            if update_data:
                await self.db.document_jobs.update_one(
                    {"_id": ObjectId(document_job_id)},
                    {"$set": update_data}
                )
                
        except Exception as e:
            self.logger.error(f"Lỗi cập nhật document job {document_job_id}: {e}")
