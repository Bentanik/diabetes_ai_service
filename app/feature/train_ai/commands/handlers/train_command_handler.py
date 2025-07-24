from bson import ObjectId
from app.database import get_collections
from core.cqrs import CommandRegistry
from core.cqrs.base import CommandHandler
from app.feature.train_ai import TrainCommand
from core.result.result import Result
from shared.messages import DocumentResult
from utils import get_logger


@CommandRegistry.register_handler(TrainCommand)
class TrainCommandHandler(CommandHandler):
    def __init__(self):
        super().__init__()
        self.logger = get_logger(__name__)

    async def execute(self, command: TrainCommand) -> Result[None]:
        self.logger.info(f"Huấn luyện mô hình RAG: {command.document_id}")

        # Validate command
        if not ObjectId.is_valid(command.document_id):
            return self._failure(DocumentResult.NOT_FOUND, command.document_id)

        # Lấy document
        collections = get_collections()
        document = await collections.documents.find_one(
            {"_id": ObjectId(command.document_id)}
        )
        if not document:
            return self._failure(DocumentResult.NOT_FOUND, command.document_id)

        # Đưa vô Train

        return Result.success(message="Huấn luyện mô hình RAG thành công")

    def _failure(self, result_msg_obj, context: str) -> Result[None]:
        self.logger.warning(f"{result_msg_obj.message}: {context}")
        return Result.failure(message=result_msg_obj.message, code=result_msg_obj.code)
