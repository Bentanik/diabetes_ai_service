from pymongo import ReturnDocument
from app.database.manager import get_collections
from app.database.models.setting_model import SettingModel
from ..update_setting_command import UpdateSettingCommand
from core.cqrs import CommandHandler, CommandRegistry
from core.result import Result
from shared.messages import SettingResult
from utils import get_logger
from bson import ObjectId


@CommandRegistry.register_handler(UpdateSettingCommand)
class UpdateSettingCommandHandler(CommandHandler):
    """
    Handler xử lý UpdateSettingCommand để cập nhật cài đặt.
    """

    def __init__(self):
        """
        Khởi tạo handler
        """
        super().__init__()
        self.logger = get_logger(__name__)
        self.collections = get_collections()

    async def execute(self, command: UpdateSettingCommand) -> Result:
        try:
            # Cập nhật number_of_passages và search_accuracy nếu có
            if command.number_of_passages is not None or command.search_accuracy is not None:
                await self.collections.settings.find_one_and_update(
                    {},
                    {"$set": {
                        "number_of_passages": command.number_of_passages,
                        "search_accuracy": command.search_accuracy,
                    }},
                    return_document=ReturnDocument.AFTER,
                )

            # Xử lý list_knowledge_id
            if command.list_knowledge_id is not None:
                if len(command.list_knowledge_id) == 0:
                    # Nếu mảng rỗng, reset select_training thành False cho tất cả knowledge
                    await self.collections.knowledges.update_many(
                        {},
                        {"$set": {"select_training": False}}
                    )
                else:
                    # Nếu có id, set select_training = True cho các id đó
                    object_ids = []
                    for knowledge_id in command.list_knowledge_id:
                        try:
                            object_ids.append(ObjectId(knowledge_id))
                        except Exception:
                            self.logger.warning(f"Invalid knowledge_id ignored: {knowledge_id}")

                    if object_ids:
                        # Trước tiên có thể reset hết select_training về False
                        await self.collections.knowledges.update_many(
                            {},
                            {"$set": {"select_training": False}}
                        )
                        # Sau đó set True cho những id được chọn
                        await self.collections.knowledges.update_many(
                            {"_id": {"$in": object_ids}},
                            {"$set": {"select_training": True}}
                        )

            return Result.success(
                code=SettingResult.UPDATED.code,
                message=SettingResult.UPDATED.message,
            )

        except Exception as e:
            self.logger.error(f"Update setting error: {e}", exc_info=True)
            return Result.failure(
                code="UPDATE_ERROR",
                message=f"Lỗi cập nhật setting: {str(e)}"
            )

