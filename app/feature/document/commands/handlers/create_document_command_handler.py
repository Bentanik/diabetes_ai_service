import asyncio
from bson import ObjectId
from app.config import MinioConfig
from app.database.manager import get_collections
from app.storage import MinioManager
from app.worker.tasks.document_jobs import DocumentJob, add_document_job
from core.cqrs import CommandHandler, CommandRegistry
from core.result import Result
from ..create_document_command import CreateDocumentCommand
from shared.messages import KnowledgeMessage, DocumentMessage
from app.database.enums import DocumentJobStatus, DocumentJobType, DocumentStatus
from app.database.value_objects import DocumentFile, ProcessingStatus
from app.database.models import DocumentJobModel
from utils import get_logger


@CommandRegistry.register_handler(CreateDocumentCommand)
class CreateDocumentCommandHandler(CommandHandler):
    def __init__(self):
        """Khởi tạo handler, logger và db"""
        super().__init__()
        self.db = get_collections()
        self.logger = get_logger(__name__)

    async def execute(self, cmd: CreateDocumentCommand) -> Result[None]:
        """
        Thực hiện toàn bộ luồng tạo tài liệu:
        1. Validate dữ liệu đầu vào (knowledge_id hợp lệ, title chưa tồn tại)
        2. Upload file lên Minio
        3. Lưu thông tin DocumentJob vào DB và đẩy job vào Queue
        4. Trả về Result.success nếu tất cả thành công
        """
        self.logger.info(f"Tạo tài liệu: {cmd.title}")

        if (res := await self._validate_command(cmd)) is not None:
            return res

        try:
            file_path = await self._upload_file(cmd)
        except Exception as e:
            return self._fail(DocumentMessage.UPLOAD_FAILED, str(e))

        await self._enqueue_document_job(cmd, file_path)

        return Result.success(message=DocumentMessage.CREATING.message,
                              code=DocumentMessage.CREATING.code)

    async def _validate_command(self, cmd: CreateDocumentCommand) -> Result | None:
        """
        Kiểm tra dữ liệu đầu vào:
        - knowledge_id phải hợp lệ
        - Knowledge tương ứng tồn tại trong DB
        - Title tài liệu chưa tồn tại
        Trả về Result.failure nếu có lỗi, hoặc None nếu hợp lệ
        """
        if not ObjectId.is_valid(cmd.knowledge_id):
            return self._fail(KnowledgeMessage.NOT_FOUND, cmd.knowledge_id)

        if not await self.db.knowledges.count_documents({"_id": ObjectId(cmd.knowledge_id)}):
            return self._fail(KnowledgeMessage.NOT_FOUND, cmd.knowledge_id)
        if await self.db.documents.count_documents({"title": cmd.title}):
            return self._fail(DocumentMessage.TITLE_EXISTS, cmd.title)

        return None

    async def _upload_file(self, cmd: CreateDocumentCommand) -> str:
        """
        Upload file tài liệu lên Minio:
        - Đọc toàn bộ nội dung file
        - Tạo object name theo format: {knowledge_id}/{filename}
        - Upload file lên bucket DOCUMENTS_BUCKET
        - Trả về đường dẫn đầy đủ trên Minio
        """
        content = await cmd.file.read()
        object_name = f"{cmd.knowledge_id}/{cmd.file.filename}"

        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: MinioManager.get_instance().upload_file(
                MinioConfig.DOCUMENTS_BUCKET,
                object_name,
                content,
                cmd.file.content_type or "application/octet-stream"
            )
        )

        return f"{MinioConfig.DOCUMENTS_BUCKET}/{object_name}"

    async def _enqueue_document_job(self, cmd: CreateDocumentCommand, file_path: str):
        """
        Tạo DocumentJob và lưu vào DB + Queue:
        1. Tạo DocumentJobModel với thông tin file, status, type, priority, v.v.
        2. Insert model vào collection document_jobs
        3. Tạo DocumentJob cho queue và gọi add_document_job
        """
        document_id = str(ObjectId())

        doc_model = DocumentJobModel(
            document_id=document_id,
            knowledge_id=cmd.knowledge_id,
            title=cmd.title,
            description=cmd.description,
            file=DocumentFile(
                path=file_path,
                size_bytes=0,
                name=cmd.file.filename,
                type=cmd.file.content_type
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

    def _fail(self, msg_obj, ctx: str) -> Result[None]:
        """
        Logger warning và trả về Result.failure
        """
        self.logger.warning(f"{msg_obj.message}: {ctx}")
        return Result.failure(message=msg_obj.message, code=msg_obj.code)
