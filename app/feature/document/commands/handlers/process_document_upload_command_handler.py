import tempfile
import os

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorCollection
from core.cqrs import CommandRegistry, CommandHandler
from core.result.result import Result
from rag import DocumentParser
from shared.messages.document_message import DocumentResult
from shared.messages.knowledge_message import KnowledgeResult
from app.database.manager import get_collections
from app.storage.minio_manager import minio_manager
from utils import FileHashUtils, get_logger
from utils.diabetes_scorer_utils import get_scorer
from app.database.models import (
    DocumentModel,
    DocumentJobStatus,
    DocumentType,
)
from app.feature.document import ProcessDocumentUploadCommand

from pymongo.errors import DuplicateKeyError


@CommandRegistry.register_handler(ProcessDocumentUploadCommand)
class ProcessDocumentUploadCommandHandler(CommandHandler):
    def __init__(self):
        self.logger = get_logger(__name__)
        self.document_parser = DocumentParser()
        self.scorer = get_scorer()

    async def execute(self, command: ProcessDocumentUploadCommand) -> Result[None]:
        self.logger.info(f"Xử lý tài liệu: {command.title}")
        collections = get_collections()

        # 1. Validate dữ liệu đầu vào
        validation_result = await self._validate_command(command)
        if validation_result is not None:
            return await self._handle_failure_with_job_update(
                validation_result,
                context=f"Validation failed for document '{command.title}'",
                document_id=command.document_id,
            )

        # 2. Cập nhật trạng thái job sang PROCESSING
        await self._update_document_job_status(
            document_id=command.document_id,
            status=DocumentJobStatus.PROCESSING,
            progress=20,
            message="Bắt đầu xử lý tài liệu",
        )

        temp_dir, temp_path = None, None
        try:
            # 3. Tải file về temp
            temp_dir, temp_path = await self._download_file_to_temp(command)

            # 4. Kiểm tra trùng lặp tài liệu dựa trên hash file
            is_duplicate = await self._check_duplicate(collections.documents, temp_path)
            if is_duplicate:
                return await self._handle_failure_with_job_update(
                    DocumentResult.DUPLICATE,
                    context=command.title,
                    document_id=command.document_id,
                )

            # # 5. Phân tích tài liệu
            documents = self.document_parser.load_document(temp_path)
            if not documents:
                return await self._handle_failure_with_job_update(
                    DocumentResult.NO_CONTENT,
                    context=command.title,
                    document_id=command.document_id,
                )

            # # 6. Tính điểm
            average_score = self._calculate_average_diabetes_score(documents)

            # # 7. Lưu vào DB
            save_result = await self._save_document_model(
                command, command.file_path, temp_path, average_score
            )
            if save_result is not None:
                return save_result

            # # 8. Cập nhật job COMPLETED
            await self._update_document_job_status(
                document_id=command.document_id,
                status=DocumentJobStatus.COMPLETED,
                progress=100,
                message="Hoàn thành xử lý tài liệu",
                priority_diabetes=average_score,
                is_diabetes=average_score > 0.5,
            )

            return Result.success(
                message=DocumentResult.CREATED.message,
                code=DocumentResult.CREATED.code,
            )

        except Exception as e:
            self.logger.error(f"Lỗi xử lý tài liệu: {e}", exc_info=True)
            return await self._handle_failure_with_job_update(
                DocumentResult.FAILED_TO_PARSE,
                context=command.title,
                document_id=command.document_id,
            )
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            if temp_dir and os.path.exists(temp_dir):
                os.rmdir(temp_dir)

    # --- Hỗ trợ ---

    async def _validate_command(
        self, command: ProcessDocumentUploadCommand
    ) -> Result | None:
        if not ObjectId.is_valid(command.knowledge_id):
            return self._failure(KnowledgeResult.NOT_FOUND, command.knowledge_id)

        collections = get_collections()

        if not await collections.knowledges.count_documents(
            {"_id": ObjectId(command.knowledge_id)}
        ):
            return self._failure(KnowledgeResult.NOT_FOUND, command.knowledge_id)

        return None

    async def _update_document_job_status(
        self,
        document_id: str,
        status: DocumentJobStatus = None,
        progress: float = None,
        message: str = None,
        progress_message: str = None,
        priority_diabetes: float = None,
        is_diabetes: bool = None,
    ):
        update_fields = {}
        if status is not None:
            update_fields["status"] = status
        if progress is not None:
            update_fields["progress"] = progress
        if message is not None:
            update_fields["progress_message"] = message
        if progress_message is not None:
            update_fields["progress_message"] = progress_message
        if priority_diabetes is not None:
            update_fields["priority_diabetes"] = priority_diabetes
        if is_diabetes is not None:
            update_fields["is_diabetes"] = is_diabetes

        if update_fields:
            collections = get_collections()
            await collections.document_jobs.update_one(
                {"document_id": document_id},
                {"$set": update_fields},
            )

    async def _download_file_to_temp(
        self, command: ProcessDocumentUploadCommand
    ) -> tuple[str, str]:
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, os.path.basename(command.file_path))

        file_path_parts = command.file_path.split("/", 1)
        bucket_name = file_path_parts[0]
        object_name = file_path_parts[1] if len(file_path_parts) > 1 else ""

        response = minio_manager.get_file(
            bucket_name=bucket_name,
            object_name=object_name,
        )
        with open(temp_path, "wb") as f:
            for chunk in response.stream(32 * 1024):
                f.write(chunk)

        return temp_dir, temp_path

    async def _check_duplicate(
        self, documents_collection: AsyncIOMotorCollection, file_path: str
    ) -> bool:
        file_hash = FileHashUtils.calculate_file_hash(file_path)
        print(f"file_hash: {file_hash}")
        existing = await documents_collection.find_one({"file_hash": file_hash})
        return existing is not None

    def _calculate_average_diabetes_score(self, documents: list) -> float:
        total = 0.0
        for doc in documents:
            total += self.scorer.calculate_diabetes_score(doc.page_content)
        return total / max(len(documents), 1)

    async def _save_document_model(
        self,
        command: ProcessDocumentUploadCommand,
        file_path: str,
        file_tmp_path: str,
        average_score: float,
    ) -> Result | None:
        file_size = os.path.getsize(file_tmp_path)
        file_hash = FileHashUtils.calculate_file_hash(file_tmp_path)
        document_model = DocumentModel(
            knowledge_id=ObjectId(command.knowledge_id),
            title=command.title,
            description=command.description,
            file_path=file_path,
            file_size_bytes=file_size,
            file_hash=file_hash,
            type=DocumentType.UPLOAD,
            priority_diabetes=average_score,
        )
        collections = get_collections()
        try:
            await collections.documents.insert_one(document_model.to_dict())
        except DuplicateKeyError:
            return await self._handle_failure_with_job_update(
                DocumentResult.DUPLICATE,
                context=command.title,
                document_id=command.document_id,
            )

    async def _handle_failure_with_job_update(
        self, result_msg_obj, context: str, document_id: str
    ) -> Result[None]:
        self.logger.warning(f"{result_msg_obj.message}: {context}")
        await self._update_document_job_status(
            document_id=document_id,
            status=DocumentJobStatus.FAILED,
            progress_message=f"{result_msg_obj.message}: {context}",
        )
        return Result.failure(message=result_msg_obj.message, code=result_msg_obj.code)

    def _failure(self, result_msg_obj, context: str) -> Result[None]:
        self.logger.warning(f"{result_msg_obj.message}: {context}")
        return Result.failure(message=result_msg_obj.message, code=result_msg_obj.code)
