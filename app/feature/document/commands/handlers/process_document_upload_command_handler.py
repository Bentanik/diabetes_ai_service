import tempfile
import os

from bson import ObjectId
from app.database.manager import get_collections
from app.database.models.document_job_model import DocumentJobStatus
from core.cqrs import CommandRegistry
from core.cqrs.base import CommandHandler
from app.feature.document import ProcessDocumentUploadCommand
from core.result.result import Result
from shared.messages.document_message import DocumentResult
from shared.messages.knowledge_message import KnowledgeResult
from utils import FileHashUtils, get_logger
from app.storage.minio_manager import minio_manager


@CommandRegistry.register_handler(ProcessDocumentUploadCommand)
class ProcessDocumentUploadCommandHandler(CommandHandler):
    def __init__(self):
        super().__init__()
        self.logger = get_logger(__name__)

    async def execute(self, command: ProcessDocumentUploadCommand) -> Result[None]:
        self.logger.info(f"Xử lý tải lên tài liệu: {command.title}")
        collections = get_collections()
        # 1. Validate command
        valid_result = await self._validate_command(command)
        if valid_result is not None:
            return valid_result

        # 2. Check duplicate
        await collections.document_jobs.update_one(
            {"document_id": command.document_id},
            {
                "$set": {
                    "status": DocumentJobStatus.PROCESSING,
                    "progress": 15,
                    "progress_message": "Đang kiểm tra tài liệu trùng lặp",
                }
            },
        )

        is_duplicate = await self._validate_diabetes_document(command)
        if is_duplicate:
            return Result.failure(
                message=DocumentResult.DUPLICATE.message,
                code=DocumentResult.DUPLICATE.code,
            )

        # 3. Kiểm tra độ trùng khớp với chủ đề đái tháo đường

        return Result.success(
            message=DocumentResult.CREATED.message,
            code=DocumentResult.CREATED.code,
        )

    async def _validate_command(
        self, command: ProcessDocumentUploadCommand
    ) -> Result | None:
        if not ObjectId.is_valid(command.knowledge_id):
            return self._failure(KnowledgeResult.NOT_FOUND, command.knowledge_id)

        collections = get_collections()

        # Kiểm tra knowledge_id có tồn tại
        if not await collections.knowledges.count_documents(
            {"_id": ObjectId(command.knowledge_id)}
        ):
            return self._failure(KnowledgeResult.NOT_FOUND, command.knowledge_id)

        # Check title tồn tại trong knowledge
        if await collections.documents.count_documents(
            {"title": command.title, "knowledge_id": command.knowledge_id}
        ):
            return self._failure(KnowledgeResult.TITLE_EXISTS, command.title)

        return None

    async def _validate_diabetes_document(
        self, command: ProcessDocumentUploadCommand
    ) -> bool:
        """
        Check file trùng dựa trên hash từ MinIO object.
        """
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, command.file_name)

        try:
            # 1. Tải file từ MinIO
            response = minio_manager.get_file(
                bucket_name=command.bucket_name,
                object_name=command.object_name,
            )

            with open(temp_path, "wb") as f:
                for chunk in response.stream(32 * 1024):
                    f.write(chunk)

            # 2. Tính file hash
            file_hash = FileHashUtils.calculate_file_hash(temp_path)

            # 3. Kiểm tra hash trùng
            collections = get_collections()
            existing = await FileHashUtils.check_duplicate_by_hash(
                collections.documents, file_hash
            )

            return existing is not None

        except Exception as e:
            self.logger.error(f"Lỗi khi kiểm tra duplicate: {e}")
            return False

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)

    def _failure(self, result_msg_obj, context: str) -> Result[None]:
        self.logger.warning(f"{result_msg_obj.message}: {context}")
        return Result.failure(message=result_msg_obj.message, code=result_msg_obj.code)
