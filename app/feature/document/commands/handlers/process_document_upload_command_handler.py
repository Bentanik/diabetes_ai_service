"""
Process Document Upload Command Handler - Xử lý command upload tài liệu

File này định nghĩa handler để xử lý ProcessDocumentUploadCommand, thực hiện việc upload tài liệu vào database và storage.
"""

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
from rag.parser import parse_file
from rag.chunking import Chunker
from shared.messages import DocumentMessage
from app.nlp.diabetes_classifier import DiabetesClassifier
from ..process_document_upload_command import ProcessDocumentUploadCommand
from core.result import Result
from utils import get_logger, FileHashUtils


@CommandRegistry.register_handler(ProcessDocumentUploadCommand)
class ProcessDocumentUploadCommandHandler(CommandHandler):
    """Handler xử lý upload tài liệu: tải file, phân tích, chia nhỏ và lưu vào database"""
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger(__name__)
        self.collections = get_collections()
        self.minio_manager = MinioManager.get_instance()
        self.chunker = Chunker()
        self.diabetes_classifier = DiabetesClassifier()
        self.file_hash_utils = FileHashUtils()

    async def execute(self, command: ProcessDocumentUploadCommand) -> Result[None]:
        """Thực thi xử lý upload tài liệu một cách bất đồng bộ"""
        # Kiểm tra đầu vào
        if not command.document_job_id or not ObjectId.is_valid(command.document_job_id):
            return Result.failure(message="ID công việc tài liệu không hợp lệ", code="INVALID_INPUT")

        self.logger.info(f"Bắt đầu xử lý tài liệu: {command.document_job_id}")
        temp_dir = None

        try:
            # Lấy thông tin document job
            document_job = await self._get_document_job(command.document_job_id)
            if not document_job:
                return Result.failure(
                    message=DocumentMessage.NOT_FOUND.message, 
                    code=DocumentMessage.NOT_FOUND.code
                )

            # Tải file về - không block thread
            temp_dir, temp_path = await self._download_file_async(command.document_job_id, document_job.file.path)

            # Kiểm tra trùng lặp
            if await self._is_duplicate_async(command.document_job_id, temp_path):
                return Result.failure(
                    message=DocumentMessage.DUPLICATE.message, 
                    code=DocumentMessage.DUPLICATE.code
                )

            # Xử lý tài liệu bất đồng bộ
            content = await self._parse_content_async(command.document_job_id, temp_path)
            chunks = await self._create_chunks_async(command.document_job_id, content)
            scores = await self._score_chunks_async(command.document_job_id, chunks)

            # Lưu vào database
            await self._save_document_async(command.document_job_id, document_job, temp_path, document_job.file.path, chunks, scores)
            await self._update_status_async(
                job_id=command.document_job_id, 
                status=DocumentJobStatus.COMPLETED, 
                progress=100,
                priority_diabetes=sum(scores) / len(scores) if scores else 0.0,
                file_size=os.path.getsize(temp_path),
                message="Hoàn tất xử lý",
            )

            self.logger.info(f"Xử lý tài liệu thành công: {command.document_job_id}")
            return Result.success(
                message=DocumentMessage.CREATED.message, 
                code=DocumentMessage.CREATED.code
            )

        except Exception as e:
            self.logger.error(f"Lỗi xử lý tài liệu {command.document_job_id}: {str(e)}")
            await self._update_status_async(
                job_id=command.document_job_id, 
                status=DocumentJobStatus.FAILED, 
                message=f"Lỗi: {str(e)}"
            )
            return Result.failure(
                message=DocumentMessage.UPLOAD_FAILED.message, 
                code=DocumentMessage.UPLOAD_FAILED.code
            )
        finally:
            # Dọn dẹp file tạm không block
            if temp_dir:
                asyncio.create_task(self._cleanup_temp_files_async(temp_dir))

    async def _get_document_job(self, job_id: str) -> DocumentJobModel:
        """Lấy thông tin document job từ database bất đồng bộ"""
        await self._update_status_async(
            job_id=job_id, 
            status=DocumentJobStatus.PROCESSING, 
            progress=15, 
            message="Đang xử lý tài liệu"
        )
        
        self.logger.info(f"Tìm kiếm thông tin công việc: {job_id}")
        data = await self.collections.document_jobs.find_one({"_id": ObjectId(job_id)})
        
        if not data:
            self.logger.warning(f"Không tìm thấy công việc tài liệu: {job_id}")
            await self._update_status_async(
                job_id=job_id, 
                status=DocumentJobStatus.FAILED, 
                message="Không tìm thấy công việc"
            )
            return None
        
        self.logger.info(f"Tìm thấy công việc tài liệu thành công: {job_id}")
        return DocumentJobModel.from_dict(data)

    async def _download_file_async(self, job_id: str, file_path: str) -> Tuple[str, str]:
        """Tải file từ MinIO về thư mục tạm một cách bất đồng bộ"""
        await self._update_status_async(
            job_id=job_id, 
            status=DocumentJobStatus.PROCESSING, 
            progress=30, 
            message="Đang tải tệp tin"
        )
        
        self.logger.info(f"Bắt đầu tải file: {file_path}")
        
        # Tạo thư mục tạm
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, os.path.basename(file_path))
        
        # Tách bucket và object path
        bucket, object_path = file_path.split("/", 1)
        
        # Sử dụng executor để không block event loop
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, 
            self.minio_manager.get_file, 
            bucket, 
            object_path
        )
        
        # Ghi file bất đồng bộ
        def write_file():
            with open(temp_path, "wb") as f:
                for chunk in response.stream(32 * 1024):
                    f.write(chunk)
        
        await loop.run_in_executor(None, write_file)
        
        self.logger.info(f"Tải file thành công: {temp_path}")
        return temp_dir, temp_path

    async def _is_duplicate_async(self, job_id: str, temp_path: str) -> bool:
        """Kiểm tra file có bị trùng lặp không bằng cách bất đồng bộ"""
        await self._update_status_async(
            job_id=job_id, 
            status=DocumentJobStatus.PROCESSING, 
            progress=40, 
            message="Kiểm tra tài liệu trùng lặp"
        )
        
        self.logger.info(f"Kiểm tra trùng lặp cho file: {temp_path}")
        
        # Tính hash bất đồng bộ
        loop = asyncio.get_event_loop()
        file_hash = await loop.run_in_executor(
            None, 
            self.file_hash_utils.calculate_file_hash, 
            temp_path
        )
        
        # Kiểm tra trong database
        count = await self.collections.documents.count_documents({"file_hash": file_hash})
        
        if count > 0:
            self.logger.warning(f"Phát hiện tài liệu trùng lặp với hash: {file_hash}")
            await self._update_status_async(
                job_id=job_id, 
                status=DocumentJobStatus.FAILED, 
                message="Tài liệu đã tồn tại trong hệ thống", 
                document_status=DocumentStatus.DUPLICATE
            )
            return True
        
        self.logger.info("Tài liệu không bị trùng lặp")
        return False

    async def _parse_content_async(self, job_id: str, temp_path: str):
        """Phân tích nội dung tài liệu bất đồng bộ"""
        await self._update_status_async(
            job_id=job_id, 
            status=DocumentJobStatus.PROCESSING, 
            progress=50, 
            message="Phân tích nội dung tài liệu"
        )
        
        self.logger.info(f"Bắt đầu phân tích nội dung: {temp_path}")
        
        # Parse file bất đồng bộ
        content = await parse_file(temp_path)
        
        self.logger.info(f"Phân tích thành công, độ dài nội dung: {len(content.content)} ký tự")
        return content

    async def _create_chunks_async(self, job_id: str, content) -> List:
        """Chia nhỏ tài liệu thành chunks bất đồng bộ"""
        await self._update_status_async(
            job_id=job_id, 
            status=DocumentJobStatus.PROCESSING, 
            progress=65, 
            message="Chia nhỏ tài liệu"
        )
        
        self.logger.info(f"Bắt đầu chia nhỏ nội dung, độ dài: {len(content.content)} ký tự")
        
        # Chunk bất đồng bộ
        chunks = await self.chunker.chunk_async(text=content.content)
        
        self.logger.info(f"Chia nhỏ thành công, số lượng chunks: {len(chunks)}")
        return chunks

    async def _score_chunks_async(self, job_id: str, chunks: List) -> List[float]:
        """Đánh giá độ liên quan đến tiểu đường bất đồng bộ"""
        await self._update_status_async(
            job_id=job_id, 
            status=DocumentJobStatus.PROCESSING, 
            progress=80, 
            message="Đánh giá độ liên quan đến bệnh tiểu đường"
        )
        
        self.logger.info(f"Bắt đầu đánh giá {len(chunks)} chunks")
        
        # Xử lý theo batch để tránh quá tải và không block
        batch_size = 50
        all_scores = []
        total_batches = (len(chunks) - 1) // batch_size + 1
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            
            # Đánh giá batch bất đồng bộ
            batch_scores = await self.diabetes_classifier.score_chunks(batch_chunks)
            all_scores.extend(batch_scores)
            
            # Cho phép các task khác chạy
            await asyncio.sleep(0)
            
            batch_num = i // batch_size + 1
            self.logger.info(f"Hoàn thành đánh giá batch {batch_num}/{total_batches}")

        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        self.logger.info(f"Hoàn tất đánh giá tất cả chunks, điểm trung bình: {avg_score:.3f}")
        return all_scores

    async def _save_document_async(self, job_id: str, document_job: DocumentJobModel, temp_path: str, file_path: str,
                                 chunks: List, scores: List[float]):
        """Lưu tài liệu và chunks vào database bất đồng bộ"""
        await self._update_status_async(
            job_id=job_id, 
            status=DocumentJobStatus.PROCESSING, 
            progress=90, 
            message="Lưu dữ liệu vào cơ sở dữ liệu"
        )
        
        self.logger.info(f"Bắt đầu lưu tài liệu và {len(chunks)} chunks")
        
        # Tính toán thông tin document
        priority = sum(scores) / len(scores) if scores else 0.0
        
        # Tính hash bất đồng bộ
        loop = asyncio.get_event_loop()
        file_hash = await loop.run_in_executor(
            None, 
            self.file_hash_utils.calculate_file_hash, 
            temp_path
        )
        
        # Tạo document model
        document = DocumentModel(
            knowledge_id=document_job.knowledge_id,
            title=document_job.title,
            description=document_job.description,
            document_type=DocumentType.UPLOADED,
            file=DocumentFile(
                path=file_path,
                size_bytes=os.path.getsize(temp_path),
                name=os.path.basename(temp_path),
                type=document_job.file.type,
            ),
            priority_diabetes=priority,
            file_hash=file_hash,
        )

        document.id = document_job.document_id
        
        # Lưu document
        await self.collections.documents.insert_one(document.to_dict())
        self.logger.info(f"Lưu document thành công: {document_job.id}")
        
        # Lưu chunks theo batch bất đồng bộ
        await self._save_chunks_in_batches_async(document_job, chunks, scores)

        # Cập nhật thống kê của knowledge
        await self._update_knowledge_stats(document_job.knowledge_id, document.file.size_bytes)
        
        self.logger.info("Lưu tất cả dữ liệu thành công")

    async def _save_chunks_in_batches_async(self, document_job: DocumentJobModel, chunks: List, 
                                          scores: List[float], batch_size: int = 500):
        """Lưu chunks theo batch bất đồng bộ"""
        total_batches = (len(chunks) - 1) // batch_size + 1
        self.logger.info(f"Lưu {len(chunks)} chunks theo {total_batches} batch")
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = []
            for j, chunk in enumerate(chunks[i:i + batch_size]):
                index = i + j
                chunk_model = DocumentChunkModel(
                    document_id=document_job.document_id,
                    knowledge_id=document_job.knowledge_id,
                    content=chunk.content,
                    diabetes_score=scores[index],
                    is_active=True
                )
                batch_chunks.append(chunk_model.to_dict())
            
            # Lưu batch
            await self.collections.document_chunks.insert_many(batch_chunks)
            
            # Cho phép các task khác chạy
            await asyncio.sleep(0)
            
            batch_num = i // batch_size + 1
            self.logger.info(f"Lưu batch {batch_num}/{total_batches} thành công")

    async def _update_knowledge_stats(self, knowledge_id: str, size_bytes: int):
        """Cập nhật thống kê của knowledge"""
        await self.collections.knowledges.update_one(
            {"_id": ObjectId(knowledge_id)},
            {"$inc": {
                "document_count": 1,
                "total_size_bytes": size_bytes
            }}
        )

    async def _update_status_async(self, job_id: str, status: DocumentJobStatus, progress: float = None, 
                                 message: str = None,
                                 priority_diabetes: float = None,
                                 document_status: DocumentStatus = DocumentStatus.NORMAL,
                                 file_size: int = None):
        """Cập nhật trạng thái document job bất đồng bộ"""
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
        """Dọn dẹp file tạm bất đồng bộ"""
        if not os.path.exists(temp_dir):
            return
            
        try:
            # Dọn dẹp bất đồng bộ
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, shutil.rmtree, temp_dir, True)
            self.logger.info(f"Dọn dẹp file tạm thành công: {temp_dir}")
        except Exception as e:
            self.logger.warning(f"Không thể dọn dẹp file tạm: {temp_dir}, lỗi: {str(e)}")
