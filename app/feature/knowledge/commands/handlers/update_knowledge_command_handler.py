from datetime import datetime
from bson import ObjectId
from pymongo import ReturnDocument

from app.database import get_collections
from app.feature.knowledge import UpdateKnowledgeCommand
from core.cqrs import CommandRegistry, CommandHandler
from core.result.result import Result
from shared.messages import KnowledgeResult
from utils import get_logger


@CommandRegistry.register_handler(UpdateKnowledgeCommand)
class UpdateKnowledgeCommandHandler(CommandHandler):
    def __init__(self):
        super().__init__()
        self.logger = get_logger(__name__)

    async def execute(self, command: UpdateKnowledgeCommand) -> Result[None]:
        """
        Cập nhật cơ sở tri thức
        Args:
            command: UpdateKnowledgeCommand

        Returns:
            Result[None]: Kết quả thành công hoặc lỗi
        """
        self.logger.info(f"Cập nhật cơ sở tri thức: {command.id}")

        if not ObjectId.is_valid(command.id):
            self.logger.warning(f"ID không hợp lệ: {command.id}")
            return Result.failure(
                message=KnowledgeResult.NOT_FOUND.message,
                code=KnowledgeResult.NOT_FOUND.code,
            )

        collection = get_collections()

        # Chỉ cập nhật các field có giá trị
        update_fields = {}
        if command.name is not None:
            update_fields["name"] = command.name
        if command.description is not None:
            update_fields["description"] = command.description
        if command.select_training is not None:
            update_fields["select_training"] = command.select_training

        if not update_fields:
            self.logger.info(f"Không có trường nào để cập nhật cho ID: {command.id}")
            return Result.success(
                message=KnowledgeResult.NO_UPDATE.message,
                code=KnowledgeResult.NO_UPDATE.code,
                data=None,
            )

        update_fields["updated_at"] = datetime.now()

        updated_doc = await collection.knowledges.find_one_and_update(
            {"_id": ObjectId(command.id)},
            {"$set": update_fields},
            return_document=ReturnDocument.AFTER,
        )

        if not updated_doc:
            self.logger.warning(f"Không tìm thấy cơ sở tri thức với id: {command.id}")
            return Result.failure(
                message=KnowledgeResult.NOT_FOUND.message,
                code=KnowledgeResult.NOT_FOUND.code,
            )

        self.logger.info(f"Cơ sở tri thức đã được cập nhật: {command.id}")

        return Result.success(
            message=KnowledgeResult.UPDATED.message,
            code=KnowledgeResult.UPDATED.code,
            data=None,
        )
