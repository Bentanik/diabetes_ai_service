import os
import tempfile
from typing import List

from bson import ObjectId
from langchain.schema import Document
from motor.motor_asyncio import AsyncIOMotorCollection
from pymongo.errors import DuplicateKeyError

from app.database.models import (
    DocumentModel,
    DocumentJobStatus,
    DocumentType,
    BBox,
    DocumentParserModel,
    Metadata,
)
from app.database.manager import get_collections
from app.feature.document import ProcessDocumentUploadCommand
from app.storage.minio_manager import minio_manager
from core.cqrs import CommandHandler, CommandRegistry
from core.result.result import Result
from rag import DocumentParser
from shared.messages.document_message import DocumentResult
from shared.messages.knowledge_message import KnowledgeResult
from utils import FileHashUtils, get_logger
from utils.diabetes_scorer_utils import get_scorer


@CommandRegistry.register_handler(ProcessDocumentUploadCommand)
class ProcessDocumentUploadCommandHandler(CommandHandler):
    def __init__(self):
        self.logger = get_logger(__name__)
        self.collections = get_collections()
        self.document_parser = DocumentParser()
        self.scorer = get_scorer()

    async def execute(self, command: ProcessDocumentUploadCommand) -> Result[None]:
        self.logger.info(f"Xử lý tài liệu: {command.title}")

        # Bước 1: Validate đầu vào
        if (validation_result := await self._validate_command(command)) is not None:
            return await self._handle_failure_with_job_update(
                validation_result,
                context=command.title,
                document_id=command.document_id,
            )

        # Bước 2: Cập nhật job đang xử lý, bắt đầu 20%
        await self._update_document_job_status(
            document_id=command.document_id,
            status=DocumentJobStatus.PROCESSING,
            progress=20,
            message="Bắt đầu xử lý tài liệu",
        )

        temp_dir, temp_path = None, None
        try:
            # Bước 3: Tải file tạm
            temp_dir, temp_path = await self._download_file_to_temp(command)
            await self._update_document_job_status(
                document_id=command.document_id,
                progress=30,
                message="Tải tài liệu tạm thành công",
            )

            # Bước 4: Check file trùng
            if await self._check_duplicate(self.collections.documents, temp_path):
                return await self._handle_failure_with_job_update(
                    DocumentResult.DUPLICATE,
                    context=command.title,
                    document_id=command.document_id,
                )
            await self._update_document_job_status(
                document_id=command.document_id,
                progress=40,
                message="Kiểm tra trùng tài liệu",
            )

            # Bước 5: Phân tích tài liệu
            documents = self.document_parser.load_document(temp_path)
            if not documents:
                return await self._handle_failure_with_job_update(
                    DocumentResult.NO_CONTENT,
                    context=command.title,
                    document_id=command.document_id,
                )
            await self._update_document_job_status(
                document_id=command.document_id,
                progress=50,
                message="Phân tích nội dung tài liệu",
            )

            # Bước 6: Mapping dữ liệu
            document_parser_models = self._to_document_parser_models(
                command.document_id, documents
            )
            print("document_parser_models:", document_parser_models[1].to_dict())
            await self._update_document_job_status(
                document_id=command.document_id,
                progress=60,
                message="Lưu các đoạn trích tài liệu",
            )

            # Bước 7: Tính điểm
            average_score = self._calculate_average_diabetes_score(documents)
            await self._update_document_job_status(
                document_id=command.document_id,
                progress=70,
                message="Tính toán điểm ưu tiên (diabetes)",
            )

            # Bước 8: Lưu document
            await self._save_document_model(
                command, command.file_path, temp_path, average_score
            )
            await self._update_document_job_status(
                document_id=command.document_id,
                progress=80,
                message="Lưu thông tin tài liệu chính",
            )

            # Bước 9: Lưu các đoạn trích
            await self._save_document_parser_models(document_parser_models)
            await self._update_document_job_status(
                document_id=command.document_id,
                progress=90,
                message="Lưu các đoạn trích tài liệu",
            )

            # Bước 10: Cập nhật job hoàn tất
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

    # ==================== SUPPORT METHODS ====================

    async def _validate_command(
        self, command: ProcessDocumentUploadCommand
    ) -> Result | None:
        if not ObjectId.is_valid(command.knowledge_id):
            return self._failure(KnowledgeResult.NOT_FOUND, command.knowledge_id)

        if not await self.collections.knowledges.count_documents(
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
        update_fields = {
            k: v
            for k, v in {
                "status": status,
                "progress": progress,
                "progress_message": message or progress_message,
                "priority_diabetes": priority_diabetes,
                "is_diabetes": is_diabetes,
            }.items()
            if v is not None
        }

        if update_fields:
            await self.collections.document_jobs.update_one(
                {"document_id": document_id},
                {"$set": update_fields},
            )

    async def _download_file_to_temp(
        self, command: ProcessDocumentUploadCommand
    ) -> tuple[str, str]:
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, os.path.basename(command.file_path))

        bucket, object_path = command.file_path.split("/", 1)
        response = minio_manager.get_file(bucket_name=bucket, object_name=object_path)

        with open(temp_path, "wb") as f:
            for chunk in response.stream(32 * 1024):
                f.write(chunk)

        return temp_dir, temp_path

    async def _check_duplicate(
        self, documents_collection: AsyncIOMotorCollection, file_path: str
    ) -> bool:
        file_hash = FileHashUtils.calculate_file_hash(file_path)
        return await documents_collection.find_one({"file_hash": file_hash}) is not None

    def _calculate_average_diabetes_score(self, documents: List[Document]) -> float:
        scores = [
            self.scorer.calculate_diabetes_score(doc.page_content) for doc in documents
        ]
        return sum(scores) / max(len(scores), 1)

    def _to_document_parser_models(
        self, document_id: str, docs: List[Document]
    ) -> List[DocumentParserModel]:
        document_id = str(document_id)
        result = []
        for doc in docs:
            metadata = doc.metadata
            bbox = metadata.get("bbox", [0, 0, 0, 0])
            document_type_str = metadata.get("document_type", DocumentType.UPLOAD.value)
            document_type = (
                DocumentType(document_type_str)
                if isinstance(document_type_str, str)
                else document_type_str
            )
            model = DocumentParserModel(
                document_id=document_id,
                content=doc.page_content,
                metadata=Metadata(
                    source=metadata.get("source", ""),
                    page=metadata.get("page", 0),
                    bbox=BBox(
                        x0=bbox[0],
                        y0=bbox[1],
                        x1=bbox[2],
                        y1=bbox[3],
                    ),
                    block_index=metadata.get("block_index"),
                    document_type=document_type,
                ),
                is_active=True,
            )
            result.append(model)
        return result

    async def _save_document_model(
        self,
        command: ProcessDocumentUploadCommand,
        file_path: str,
        file_tmp_path: str,
        average_score: float,
    ):
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
        document_model.id = ObjectId(command.document_id)
        try:
            await self.collections.documents.insert_one(document_model.to_dict())
        except DuplicateKeyError:
            raise Exception("Document already exists")

    async def _save_document_parser_models(
        self, document_parser_models: List[DocumentParserModel]
    ):
        documents_parser = [model.to_dict() for model in document_parser_models]
        await self.collections.documents_parsers.insert_many(documents_parser)

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
