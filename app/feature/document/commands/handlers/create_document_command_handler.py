import asyncio
from bson import ObjectId
from app.database.manager import get_collections
from app.storage import minio_manager
from app.worker import add_document_job, DocumentJob
from core.cqrs import CommandRegistry
from core.cqrs.base import CommandHandler
from app.feature.document import CreateDocumentCommand
from core.result.result import Result
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

        # 2. Upload file lên Minio
        try:
            file_path = await self._upload_file_to_minio(command)
        except Exception as e:
            self.logger.error(f"Upload file thất bại: {e}")
            return self._failure(DocumentResult.UPLOAD_FAILED, str(e))

        # 3. Đẩy job xử lý lên queue
        await self._enqueue_document_job(command, file_path)

        return Result.success(
            message=DocumentResult.CREATING.message, code=DocumentResult.CREATING.code
        )

    async def _validate_command(self, command: CreateDocumentCommand) -> Result | None:
        if not ObjectId.is_valid(command.knowledge_id):
            return self._failure(KnowledgeResult.NOT_FOUND, command.knowledge_id)

        collections = get_collections()

        if not await collections.knowledges.count_documents(
            {"_id": ObjectId(command.knowledge_id)}
        ):
            return self._failure(KnowledgeResult.NOT_FOUND, command.knowledge_id)

        if await collections.documents.count_documents(
            {"title": command.title, "knowledge_id": command.knowledge_id}
        ):
            return self._failure(KnowledgeResult.TITLE_EXISTS, command.title)

        return None

    async def _upload_file_to_minio(self, command: CreateDocumentCommand) -> str:
        file = command.file
        file_content = await file.read()
        bucket_name = "documents"
        object_name = f"{command.knowledge_id}/{file.filename}"

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: minio_manager.upload_file(
                bucket_name,
                object_name,
                file_content,
                file.content_type or "application/octet-stream",
            ),
        )

        # Trả về đường dẫn file trên Minio
        return f"{bucket_name}/{object_name}"

    async def _enqueue_document_job(
        self, command: CreateDocumentCommand, file_path: str
    ):
        document_job = DocumentJob(
            knowledge_id=command.knowledge_id,
            title=command.title,
            description=command.description,
            file_path=file_path,
        )
        await add_document_job(document_job)

    def _failure(self, result_msg_obj, context: str) -> Result[None]:
        self.logger.warning(f"{result_msg_obj.message}: {context}")
        return Result.failure(message=result_msg_obj.message, code=result_msg_obj.code)
