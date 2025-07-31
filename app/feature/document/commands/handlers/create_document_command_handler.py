"""
Create Document Command Handler - Xử lý command tạo tài liệu

File này định nghĩa handler để xử lý CreateDocumentCommand, thực hiện việc
tạo tài liệu mới vào database và storage.
"""

import asyncio
from bson import ObjectId
from app.config import MinioConfig
from app.database import DBCollections
from app.database.enums import DocumentJobStatus, DocumentJobType
from app.database.manager import get_collections
from app.database.models import DocumentJobModel
from app.database.value_objects.processing_status import ProcessingStatus
from app.storage import MinioManager

from app.worker.tasks.document_jobs import DocumentJob, add_document_job
from core.cqrs import CommandHandler, CommandRegistry
from core.result import Result

from ..create_document_command import CreateDocumentCommand
from shared.messages.document_message import DocumentResult
from shared.messages.knowledge_message import KnowledgeResult
from utils import get_logger


@CommandRegistry.register_handler(CreateDocumentCommand)
class CreateDocumentCommandHandler(CommandHandler):
    def __init__(self):
        super().__init__()
        self.logger = get_logger(__name__)

    async def execute(self, command: CreateDocumentCommand) -> Result[None]:
        self.logger.info(f"Tạo tài liệu mới: {command.title}")

        # 1. Validate command
        valid_result = await self._validate_command(command)
        if valid_result is not None:
            return valid_result

        collections = get_collections()

        # 2. Upload file lên Minio
        try:
            file_path = await self._upload_file_to_minio(command)
        except Exception as e:
            self.logger.error(f"Upload file thất bại: {e}")
            return self._failure(DocumentResult.UPLOAD_FAILED, str(e))

        # 3. Đẩy job xử lý lên queue và lưu vào DB
        await self._enqueue_document_job(collections, command, file_path)

        return Result.success(
            message=DocumentResult.CREATING.message, code=DocumentResult.CREATING.code
        )

    async def _validate_command(self, command: CreateDocumentCommand) -> Result | None:
        if not ObjectId.is_valid(command.knowledge_id):
            return self._failure(KnowledgeResult.NOT_FOUND, command.knowledge_id)

        collections = get_collections()

        if (
            await collections.knowledges.count_documents(
                {"_id": ObjectId(command.knowledge_id)}
            )
            == 0
        ):
            return self._failure(KnowledgeResult.NOT_FOUND, command.knowledge_id)

        if await collections.documents.count_documents({"title": command.title}) > 0:
            return self._failure(DocumentResult.TITLE_EXISTS, command.title)

        return None

    async def _upload_file_to_minio(self, command: CreateDocumentCommand) -> str:
        file = command.file
        file_content = await file.read()
        bucket_name = MinioConfig.DOCUMENTS_BUCKET
        object_name = f"{command.knowledge_id}/{file.filename}"

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: MinioManager.get_instance().upload_file(
                bucket_name,
                object_name,
                file_content,
                file.content_type or "application/octet-stream",
            ),
        )

        # Trả về đường dẫn file trên Minio
        return f"{bucket_name}/{object_name}"

    async def _enqueue_document_job(
        self, db: DBCollections, command: CreateDocumentCommand, file_path: str
    ):
        # Tạo document id mới
        document_id = str(ObjectId())

        # Tạo DocumentJobModel để lưu vào DB
        processing_status = ProcessingStatus(
            status=DocumentJobStatus.PENDING,
            progress=10,
            progress_message="Đang tạo tài liệu",
        )

        document_job_model = DocumentJobModel(
            document_id=document_id,
            knowledge_id=command.knowledge_id,
            title=command.title,
            description=command.description,
            file_path=file_path,
            status=processing_status,
            type=DocumentJobType.UPLOAD,
            priority_diabetes=0,
        )
        await db.document_jobs.insert_one(document_job_model.to_dict())

        # Tạo DocumentJob để đẩy vào Redis queue
        redis_job = DocumentJob(
            id=document_job_model.id,
            type="upload_document",
        )
        await add_document_job(redis_job)

    def _failure(self, result_msg_obj, context: str) -> Result[None]:
        self.logger.warning(f"{result_msg_obj.message}: {context}")
        return Result.failure(message=result_msg_obj.message, code=result_msg_obj.code)
