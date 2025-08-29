import asyncio
from bson import ObjectId
from typing import List, Dict

from fastapi import UploadFile
from app.config import MinioConfig
from app.database.manager import get_collections
from app.storage import MinioManager
from app.worker.tasks.document_jobs import DocumentJob, add_document_job
from core.cqrs import CommandHandler, CommandRegistry
from core.result import Result
from ..create_documents_command import CreateDocumentsCommand
from shared.messages import KnowledgeMessage, DocumentMessage
from app.database.enums import DocumentJobStatus, DocumentJobType, DocumentStatus
from app.database.value_objects import DocumentFile, ProcessingStatus
from app.database.models import DocumentJobModel
from utils import get_logger


@CommandRegistry.register_handler(CreateDocumentsCommand)
class CreateDocumentsCommandHandler(CommandHandler):
    def __init__(self):
        """Khởi tạo handler, logger và db"""
        super().__init__()
        self.db = get_collections()
        self.logger = get_logger(__name__)

    async def execute(self, cmd: CreateDocumentsCommand) -> Result[List[Dict]]:
        """
        Thực hiện toàn bộ luồng tạo nhiều tài liệu:
        1. Validate dữ liệu đầu vào (knowledge_id hợp lệ, titles chưa tồn tại)
        2. Upload các file lên Minio (concurrently nếu có thể)
        3. Lưu thông tin DocumentJob vào DB và đẩy job vào Queue cho từng tài liệu (bây giờ concurrent)
        4. Trả về Result.success với danh sách kết quả từng tài liệu
        """
        self.logger.info(f"Tạo {len(cmd.files)} tài liệu cho knowledge_id: {cmd.knowledge_id}")

        if (res := await self._validate_command(cmd)) is not None:
            return res

        # Upload files concurrently (unchanged)
        upload_tasks = []
        for file, title, description in zip(cmd.files, cmd.titles, cmd.descriptions):
            upload_tasks.append(self._upload_file(file, cmd.knowledge_id))

        try:
            file_paths = await asyncio.gather(*upload_tasks)
        except Exception as e:
            return self._fail(DocumentMessage.UPLOAD_FAILED, str(e))

        # Enqueue jobs concurrently
        enqueue_tasks = []
        for file, title, description, file_path in zip(cmd.files, cmd.titles, cmd.descriptions, file_paths):
            enqueue_tasks.append(self._enqueue_document_job(cmd.knowledge_id, title, description, file, file_path))

        # Gather all enqueue tasks, allowing exceptions to be caught without stopping
        enqueue_results = await asyncio.gather(*enqueue_tasks, return_exceptions=True)

        results = []
        for idx, res in enumerate(enqueue_results):
            title = cmd.titles[idx]  # Get title for error reporting
            if isinstance(res, Exception):
                self.logger.error(f"Lỗi tạo tài liệu '{title}': {str(res)}")
                results.append({
                    "title": title,
                    "success": False,
                    "message": f"Tạo thất bại: {str(res)}"
                })
            else:
                document_id = res  # res is the document_id if successful
                results.append({
                    "document_id": document_id,
                    "title": title,
                    "message": DocumentMessage.CREATING.message,
                    "code": DocumentMessage.CREATING.code
                })

        success_count = sum(1 for r in results if "success" not in r or r.get("success", True))
        if success_count == len(cmd.files):
            return Result.success(
                message=f"Đã tạo thành công {success_count} tài liệu",
                data=results
            )
        else:
            return Result.failure(
                message=f"Đã tạo {success_count}/{len(cmd.files)} tài liệu, có lỗi xảy ra",
                data=results,
                code=400  # Hoặc mã lỗi phù hợp
            )

    async def _validate_command(self, cmd: CreateDocumentsCommand) -> Result | None:
        """
        Kiểm tra dữ liệu đầu vào:
        - knowledge_id phải hợp lệ
        - Knowledge tương ứng tồn tại trong DB
        - Các title tài liệu chưa tồn tại trong DB
        Trả về Result.failure nếu có lỗi, hoặc None nếu hợp lệ
        """
        if not ObjectId.is_valid(cmd.knowledge_id):
            return self._fail(KnowledgeMessage.NOT_FOUND, cmd.knowledge_id)

        if not await self.db.knowledges.count_documents({"_id": ObjectId(cmd.knowledge_id)}):
            return self._fail(KnowledgeMessage.NOT_FOUND, cmd.knowledge_id)

        existing_titles = set()
        for title in cmd.titles:
            if await self.db.documents.count_documents({"title": title}):
                return self._fail(DocumentMessage.TITLE_EXISTS, title)
            if title in existing_titles:
                return self._fail(DocumentMessage.TITLE_EXISTS, f"Tiêu đề trùng lặp trong batch: {title}")
            existing_titles.add(title)

        return None

    async def _upload_file(self, file: UploadFile, knowledge_id: str) -> str:
        """
        Upload file tài liệu lên Minio:
        - Đọc toàn bộ nội dung file
        - Tạo object name theo format: {knowledge_id}/{filename}
        - Upload file lên bucket DOCUMENTS_BUCKET
        - Trả về đường dẫn đầy đủ trên Minio
        """
        content = await file.read()
        object_name = f"{knowledge_id}/{file.filename}"

        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: MinioManager.get_instance().upload_file(
                MinioConfig.DOCUMENTS_BUCKET,
                object_name,
                content,
                file.content_type or "application/octet-stream"
            )
        )

        return f"{MinioConfig.DOCUMENTS_BUCKET}/{object_name}"

    async def _enqueue_document_job(self, knowledge_id: str, title: str, description: str, file: UploadFile, file_path: str) -> str:
        """
        Tạo DocumentJob và lưu vào DB + Queue:
        1. Tạo DocumentJobModel với thông tin file, status, type, priority, v.v.
        2. Insert model vào collection document_jobs
        3. Tạo DocumentJob cho queue và gọi add_document_job
        4. Trả về document_id
        """
        document_id = str(ObjectId())

        doc_model = DocumentJobModel(
            document_id=document_id,
            knowledge_id=knowledge_id,
            title=title,
            description=description,
            file=DocumentFile(
                path=file_path,
                size_bytes=0,
                name=file.filename,
                type=file.content_type
            ),
            processing_status=ProcessingStatus(
                status=DocumentJobStatus.PENDING,
                progress=10,
                progress_message="Đang tạo tài liệu"
            ),
            type=DocumentJobType.UPLOAD,
            priority_diabetes=0,
            document_status=DocumentStatus.NORMAL,
        )

        await self.db.document_jobs.insert_one(doc_model.to_dict())
        await add_document_job(DocumentJob(id=doc_model.id, type="upload_document"))

        return document_id

    def _fail(self, msg_obj, ctx: str) -> Result[None]:
        """
        Logger warning và trả về Result.failure
        """
        self.logger.warning(f"{msg_obj.message}: {ctx}")
        return Result.failure(message=msg_obj.message, code=msg_obj.code)
