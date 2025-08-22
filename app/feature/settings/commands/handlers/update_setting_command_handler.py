from pymongo import ReturnDocument
from app.database.manager import get_collections
from ..update_setting_command import UpdateSettingCommand
from core.cqrs import CommandHandler, CommandRegistry
from core.result import Result
from shared.messages import SettingMessage
from utils import get_logger
from bson import ObjectId


@CommandRegistry.register_handler(UpdateSettingCommand)
class UpdateSettingCommandHandler(CommandHandler):
    """
    Handler xử lý UpdateSettingCommand để cập nhật cài đặt.
    """

    def __init__(self):
        super().__init__()
        self.logger = get_logger(__name__)
        self.collections = get_collections()

    async def execute(self, command: UpdateSettingCommand) -> Result:
        try:
            update_fields = {}

            # Cập nhật các trường trong collection settings nếu có giá trị
            if command.top_k is not None:
                update_fields["top_k"] = command.top_k
            if command.temperature is not None:
                update_fields["temperature"] = command.temperature
            if command.max_tokens is not None:
                update_fields["max_tokens"] = command.max_tokens
            if command.search_accuracy is not None:
                update_fields["search_accuracy"] = command.search_accuracy
            if command.list_knowledge_ids is not None:
                update_fields["list_knowledge_ids"] = command.list_knowledge_ids

            if update_fields:
                await self.collections.settings.find_one_and_update(
                    {}, {"$set": update_fields}, return_document=ReturnDocument.AFTER
                )

            if command.list_knowledge_ids is not None:
                if len(command.list_knowledge_ids) == 0:
                    # Reset select_training thành False cho tất cả knowledge nếu mảng rỗng
                    await self.collections.knowledges.update_many(
                        {}, {"$set": {"select_training": False}}
                    )
                else:
                    object_ids = []
                    for knowledge_id in command.list_knowledge_ids:
                        try:
                            object_ids.append(ObjectId(knowledge_id))
                        except Exception:
                            self.logger.warning(f"Invalid knowledge_id ignored: {knowledge_id}")

                    if object_ids:
                        # Reset hết về False trước
                        await self.collections.knowledges.update_many(
                            {}, {"$set": {"select_training": False}}
                        )
                        # Set True cho các knowledge được chọn
                        await self.collections.knowledges.update_many(
                            {"_id": {"$in": object_ids}},
                            {"$set": {"select_training": True}},
                        )

            return Result.success(
                code=SettingMessage.UPDATED.code,
                message=SettingMessage.UPDATED.message,
            )

        except Exception as e:
            self.logger.error(f"Update setting error: {e}", exc_info=True)
            return Result.failure(
                code="UPDATE_ERROR",
                message=f"Lỗi cập nhật setting: {str(e)}",
            )
