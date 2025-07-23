from bson import ObjectId

from app.database.manager import get_collections
from app.feature.knowledge import DeleteKnowledgeCommand
from core.cqrs import CommandRegistry
from core.cqrs.base import CommandHandler
from core.result.result import Result
from shared.messages.knowledge_message import KnowledgeResult
from utils import get_logger


@CommandRegistry.register_handler(DeleteKnowledgeCommand)
class DeleteKnowledgeCommandHandler(CommandHandler):
    def __init__(self):
        super().__init__()
        self.logger = get_logger(__name__)

    async def execute(self, command: DeleteKnowledgeCommand) -> Result[None]:
        """
        Xóa cơ sở tri thức
        Args:
            command: DeleteKnowledgeCommand

        Returns:
            Result[None]: Kết quả thành công hoặc lỗi
        """
        self.logger.info(f"Xóa cơ sở tri thức: {command.id}")

        if not ObjectId.is_valid(command.id):
            self.logger.warning(f"ID không hợp lệ: {command.id}")
            return Result.failure(
                message=KnowledgeResult.NOT_FOUND.message,
                code=KnowledgeResult.NOT_FOUND.code,
            )

        collection = get_collections()

        # Thực hiện xóa bản ghi theo _id
        delete_result = await collection.knowledges.delete_one(
            {"_id": ObjectId(command.id)}
        )

        if delete_result.deleted_count == 0:
            self.logger.warning(f"Không tìm thấy cơ sở tri thức với id: {command.id}")
            return Result.failure(
                message=KnowledgeResult.NOT_FOUND.message,
                code=KnowledgeResult.NOT_FOUND.code,
            )

        self.logger.info(f"Cơ sở tri thức đã được xóa: {command.id}")

        return Result.success(
            message=KnowledgeResult.DELETED.message,
            code=KnowledgeResult.DELETED.code,
            data=None,
        )
