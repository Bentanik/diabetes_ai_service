from this import d

from pymongo import ReturnDocument
from app.database.manager import get_collections
from app.database.models.setting_model import SettingModel
from ..update_setting_command import UpdateSettingCommand
from core.cqrs import CommandHandler, CommandRegistry
from core.result import Result
from shared.messages import SettingResult
from utils import get_logger


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
            update_data = {}

            if command.number_of_passages is not None:
                update_data["number_of_passages"] = command.number_of_passages

            if command.search_accuracy is not None:
                update_data["search_accuracy"] = command.search_accuracy

            if (
                hasattr(command, "list_knowledge_id")
                and command.list_knowledge_id is not None
            ):
                update_data["list_knowledge_id"] = command.list_knowledge_id

            if not update_data:
                return Result.failure(
                    code="NO_DATA", message="Không có dữ liệu để cập nhật"
                )

            result = await self.collections.settings.find_one_and_update(
                {},
                {"$set": update_data},
                return_document=ReturnDocument.AFTER,
            )

            return Result.success(
                code=SettingResult.UPDATED.code,
                message=SettingResult.UPDATED.message,
                data=result,
            )

        except Exception as e:
            self.logger.error(f"Update setting error: {e}")
            return Result.failure(
                code="UPDATE_ERROR", message=f"Lỗi cập nhật setting: {str(e)}"
            )
