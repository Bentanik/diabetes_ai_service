import os
import tempfile
import shutil
import asyncio
from typing import List, Tuple
from app.database import get_collections
from app.database.enums import DocumentJobStatus, DocumentType, DocumentStatus
from app.database.models import DocumentJobModel, DocumentModel, DocumentChunkModel
from app.database.value_objects import DocumentFile
from app.storage import MinioManager
from core.cqrs import CommandHandler, CommandRegistry
from bson import ObjectId
from core.embedding import EmbeddingModel
from rag.parser import parse_file
from rag.chunking import Chunker
from shared.messages import DocumentMessage
from app.nlp.diabetes_classifier import DiabetesClassifier
from ..process_document_upload_command import ProcessDocumentUploadCommand
from core.result import Result
from utils import get_logger, FileHashUtils


@CommandRegistry.register_handler(ProcessDocumentUploadCommand)
class ProcessDocumentUploadCommandHandler(CommandHandler):

    def __init__(self):
        super().__init__()
        self.logger = get_logger(__name__)
        self.collections = get_collections()
        self.minio_manager = MinioManager.get_instance()
        self.diabetes_classifier = DiabetesClassifier()
        self.file_hash_utils = FileHashUtils()

    async def execute(self, command: ProcessDocumentUploadCommand) -> Result[None]:
        """Thực thi xử lý upload tài liệu một cách bất đồng bộ"""
        if not command.document_job_id or not ObjectId.is_valid(command.document_job_id):
            return Result.failure(message="ID công việc tài liệu không hợp lệ", code="INVALID_INPUT")

        job_id = command.document_job_id
        self.logger.info(f"Bắt đầu xử lý tài liệu: {job_id}")

        existing_job = await self.collections.document_jobs.find_one({"_id": ObjectId(job_id)})
        if not existing_job:
            return Result.failure(message=DocumentMessage.NOT_FOUND.message, code=DocumentMessage.NOT_FOUND.code)

        status = existing_job.get("processing_status", {}).get("status")
        if status in [DocumentJobStatus.PROCESSING, DocumentJobStatus.COMPLETED, DocumentJobStatus.FAILED]:
            self.logger.warning(f"Job đã ở trạng thái {status}, bỏ qua: {job_id}")
            return Result.failure(message="Job đã được xử lý", code="ALREADY_PROCESSED")

        temp_dir = None
        try:
            embedding_model = await EmbeddingModel.get_instance()
            self.chunker = Chunker(
                embedding_model=embedding_model,
                tokenizer=embedding_model.model.tokenizer,
                max_tokens=512,
                min_tokens=100,
                overlap_tokens=64,
                similarity_threshold=0.6,
            )

            # Lấy thông tin document job
            document_job = await self._get_document_job(job_id)
            if not document_job:
                return Result.failure(
                    message=DocumentMessage.NOT_FOUND.message,
                    code=DocumentMessage.NOT_FOUND.code
                )

            # Tải file về
            temp_dir, temp_path = await self._download_file_async(job_id, document_job.file.path)

            # Kiểm tra trùng file bằng hash
            if await self._is_duplicate_async(job_id, temp_path):
                return Result.failure(
                    message=DocumentMessage.DUPLICATE.message,
                    code=DocumentMessage.DUPLICATE.code
                )

            # Xử lý nội dung
            content = await self._parse_content_async(job_id, temp_path)
            chunks = await self._create_chunks_async(job_id, content)
            scores = await self._score_chunks_async(job_id, chunks)

            # Lưu document và chunks (không xóa cũ)
            await self._save_document_async(job_id, document_job, temp_path, chunks, scores)

            # Cập nhật trạng thái thành công
            await self._update_status_async(
                job_id=job_id,
                status=DocumentJobStatus.COMPLETED,
                progress=100,
                priority_diabetes=sum(scores) / len(scores) if scores else 0.0,
                file_size=os.path.getsize(temp_path),
                message="Hoàn tất xử lý",
            )

            self.logger.info(f"Xử lý tài liệu thành công: {job_id}")
            return Result.success(message=DocumentMessage.CREATED.message, code=DocumentMessage.CREATED.code)

        except Exception as e:
            self.logger.error(f"Lỗi xử lý tài liệu {job_id}: {str(e)}", exc_info=True)
            await self._update_status_async(
                job_id=job_id,
                status=DocumentJobStatus.FAILED,
                message=f"Lỗi: {str(e)}"
            )
            return Result.failure(message=DocumentMessage.UPLOAD_FAILED.message, code=DocumentMessage.UPLOAD_FAILED.code)
        finally:
            if temp_dir:
                asyncio.create_task(self._cleanup_temp_files_async(temp_dir))

    async def _get_document_job(self, job_id: str) -> DocumentJobModel:
        """Lấy thông tin document job từ DB"""
        await self._update_status_async(
            job_id=job_id,
            status=DocumentJobStatus.PROCESSING,
            progress=15,
            message="Đang xử lý tài liệu"
        )
        data = await self.collections.document_jobs.find_one({"_id": ObjectId(job_id)})
        return DocumentJobModel.from_dict(data) if data else None

    async def _download_file_async(self, job_id: str, file_path: str) -> Tuple[str, str]:
        """Tải file từ MinIO về tạm"""
        await self._update_status_async(job_id=job_id, status=DocumentJobStatus.PROCESSING, progress=30, message="Đang tải tệp tin")
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, os.path.basename(file_path))
        bucket, object_path = file_path.split("/", 1)

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, self.minio_manager.get_file, bucket, object_path)

        def write_file():
            with open(temp_path, "wb") as f:
                for chunk in response.stream(32 * 1024):
                    f.write(chunk)

        await loop.run_in_executor(None, write_file)
        return temp_dir, temp_path

    async def _is_duplicate_async(self, job_id: str, temp_path: str) -> bool:
        """Kiểm tra trùng lặp bằng hash"""
        await self._update_status_async(job_id=job_id, status=DocumentJobStatus.PROCESSING, progress=40, message="Kiểm tra trùng lặp")
        loop = asyncio.get_event_loop()
        file_hash = await loop.run_in_executor(None, self.file_hash_utils.calculate_file_hash, temp_path)
        count = await self.collections.documents.count_documents({"file_hash": file_hash})
        if count > 0:
            await self._update_status_async(
                job_id=job_id,
                status=DocumentJobStatus.FAILED,
                message="Tài liệu đã tồn tại",
                document_status=DocumentStatus.DUPLICATE
            )
            return True
        return False

    async def _parse_content_async(self, job_id: str, temp_path: str):
        """Parse nội dung file"""
        await self._update_status_async(job_id=job_id, status=DocumentJobStatus.PROCESSING, progress=50, message="Phân tích nội dung")
        return await parse_file(temp_path)

    async def _create_chunks_async(self, job_id: str, content) -> List:
        """Chia nhỏ nội dung"""
        await self._update_status_async(job_id=job_id, status=DocumentJobStatus.PROCESSING, progress=65, message="Chia nhỏ tài liệu")
        chunks = await self.chunker.chunk_async(parsed_content=content)
        return chunks

    async def _score_chunks_async(self, job_id: str, chunks: List) -> List[float]:
        """Đánh điểm tiểu đường theo batch"""
        await self._update_status_async(job_id=job_id, status=DocumentJobStatus.PROCESSING, progress=80, message="Đánh giá độ liên quan")
        batch_size = 50
        all_scores = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            scores = await self.diabetes_classifier.score_chunks(batch)
            all_scores.extend(scores)
            await asyncio.sleep(0)
        return all_scores

    async def _save_document_async(self, job_id: str, document_job: DocumentJobModel, temp_path: str,
                                 chunks: List, scores: List[float]):
        """Lưu document và chunks (không xóa cũ)"""
        await self._update_status_async(
            job_id=job_id,
            status=DocumentJobStatus.PROCESSING,
            progress=90,
            message="Lưu dữ liệu vào cơ sở dữ liệu"
        )

        # Tính hash
        loop = asyncio.get_event_loop()
        file_hash = await loop.run_in_executor(None, self.file_hash_utils.calculate_file_hash, temp_path)

        # Cập nhật hoặc tạo mới document (không xóa cũ)
        document = DocumentModel(
            knowledge_id=document_job.knowledge_id,
            title=document_job.title,
            description=document_job.description,
            document_type=DocumentType.UPLOADED,
            file=DocumentFile(
                path=document_job.file.path,
                size_bytes=os.path.getsize(temp_path),
                name=os.path.basename(temp_path),
                type=document_job.file.type,
            ),
            priority_diabetes=sum(scores) / len(scores) if scores else 0.0,
            file_hash=file_hash,
        )
        document.id = document_job.document_id

        # Upsert document
        await self.collections.documents.replace_one(
            {"_id": ObjectId(document.id)},
            document.to_dict(),
            upsert=True
        )

        await self._save_chunks_in_batches_async(document_job, chunks, scores)

        # Cập nhật thống kê
        await self._update_knowledge_stats(document_job.knowledge_id, document.file.size_bytes)

    async def _save_chunks_in_batches_async(self, document_job: DocumentJobModel, chunks: List, scores: List[float], batch_size: int = 500):
        """Lưu chunks theo batch, không xóa cũ, mỗi chunk có _id mới"""
        total = len(chunks)
        self.logger.info(f"💾 Bắt đầu lưu {total} chunks mới (giữ chunk cũ)")

        for i in range(0, total, batch_size):
            batch = chunks[i:i + batch_size]
            batch_dicts = []

            for j, chunk in enumerate(batch):
                index = i + j
                chunk_model = DocumentChunkModel(
                    document_id=document_job.document_id,
                    knowledge_id=document_job.knowledge_id,
                    content=chunk.content,
                    diabetes_score=scores[index],
                    is_active=True
                )
                batch_dicts.append(chunk_model.to_dict())  # dict không có _id

            try:
                result = await self.collections.document_chunks.insert_many(batch_dicts, ordered=False)
                self.logger.info(f"Lưu batch {i//batch_size + 1}: {len(result.inserted_ids)} chunks")
            except Exception as e:
                self.logger.warning(f"Lỗi khi lưu batch: {str(e)}")

    async def _update_knowledge_stats(self, knowledge_id: str, size_bytes: int):
        """Cập nhật thống kê knowledge"""
        await self.collections.knowledges.update_one(
            {"_id": ObjectId(knowledge_id)},
            {"$inc": {"document_count": 1, "total_size_bytes": size_bytes}}
        )

    async def _update_status_async(self, job_id: str, status: DocumentJobStatus, progress: float = None,
                                 message: str = None, priority_diabetes: float = None,
                                 document_status: DocumentStatus = DocumentStatus.NORMAL, file_size: int = None):
        """Cập nhật trạng thái job"""
        update_fields = {
            "processing_status.status": status,
            "document_status": document_status
        }
        if progress is not None:
            update_fields["processing_status.progress"] = progress
        if message is not None:
            update_fields["processing_status.progress_message"] = message
        if priority_diabetes is not None:
            update_fields["priority_diabetes"] = priority_diabetes
        if file_size is not None:
            update_fields["file.file_size_bytes"] = file_size

        await self.collections.document_jobs.update_one(
            {"_id": ObjectId(job_id)},
            {"$set": update_fields}
        )

    async def _cleanup_temp_files_async(self, temp_dir: str):
        """Dọn dẹp file tạm"""
        if os.path.exists(temp_dir):
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, shutil.rmtree, temp_dir, True)
            self.logger.info(f"Đã dọn dẹp thư mục tạm: {temp_dir}")