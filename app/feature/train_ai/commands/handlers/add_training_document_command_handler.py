from app.database import get_collections, DBCollections
from app.database.enums import DocumentJobStatus, DocumentJobType
from app.database.models import DocumentJobModel, DocumentModel
from app.database.value_objects import ProcessingStatus
from app.worker.tasks import DocumentJob, add_document_job
from core.cqrs import CommandHandler, CommandRegistry
from core.result import Result
from ..add_training_document_command import AddTrainingDocumentCommand
from shared.messages import DocumentResult
from utils import get_logger
from bson import ObjectId


@CommandRegistry.register_handler(AddTrainingDocumentCommand)
class AddTrainingDocumentCommandHandler(CommandHandler):
    def __init__(self):
        super().__init__()
        self.logger = get_logger(__name__)
        self.collections = get_collections()

    async def execute(self, command: AddTrainingDocumentCommand) -> Result[None]:
        self.logger.info(f"Thêm vào hàng đợi: {command.document_id}")

        try:
            if not ObjectId.is_valid(command.document_id):
                self.logger.info(f"ID tài liệu không hợp lệ: {command.document_id}")
                return Result.failure(
                    message=DocumentResult.NOT_FOUND.message,
                    code=DocumentResult.NOT_FOUND.code,
                )

            document_job_exists = await self.collections.document_jobs.count_documents(
                {
                    "document_id": command.document_id,
                    "type": DocumentJobType.TRAINING.value,
                }
            )

            if document_job_exists > 0:
                self.logger.info(f"Tài liệu đã được huấn luyện: {command.document_id}")
                return Result.success(
                    message=DocumentResult.TRAINING_ALREADY_EXISTS.message,
                    code=DocumentResult.TRAINING_ALREADY_EXISTS.code,
                )

            # 1. Lấy document từ MongoDB
            document = await self.collections.documents.find_one(
                {"_id": ObjectId(command.document_id)}
            )

            if not document:
                self.logger.info(
                    f"Không tìm thấy document với ID: {command.document_id}"
                )
                return Result.failure(
                    message=DocumentResult.NOT_FOUND.message,
                    code=DocumentResult.NOT_FOUND.code,
                )

            document = DocumentModel.from_dict(document)

            # 2. Thêm vào hàng đợi
            await self._enqueue_document_job(self.collections, document)

            return Result.success(
                message=DocumentResult.TRAINING_STARTED.message,
                code=DocumentResult.TRAINING_STARTED.code,
            )

        except Exception as e:
            self.logger.error(f"Lỗi xử lý tài liệu huấn luyện: {e}", exc_info=True)
            return Result.failure(
                message=DocumentResult.TRAINING_FAILED.message,
                code=DocumentResult.TRAINING_FAILED.code,
            )

    async def _enqueue_document_job(self, db: DBCollections, document: DocumentModel):
        # Tạo DocumentJobModel để lưu vào DB
        processing_status = ProcessingStatus(
            status=DocumentJobStatus.PENDING,
            progress=10,
            progress_message="Đang tạo tài liệu",
        )

        document_job_model = DocumentJobModel(
            document_id=document.id,
            knowledge_id=document.knowledge_id,
            title=document.title,
            description=document.description,
            file_path=document.file.path,
            status=processing_status,
            type=DocumentJobType.TRAINING,
            priority_diabetes=document.priority_diabetes,
        )
        await db.document_jobs.insert_one(document_job_model.to_dict())

        # Tạo DocumentJob để đẩy vào Redis queue
        redis_job = DocumentJob(
            id=document_job_model.id,
            type="training_document",
        )
        await add_document_job(redis_job)
