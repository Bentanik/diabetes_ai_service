from typing import Any

from app.database.manager import get_collections
from app.database.models.setting_model import SettingModel
from ..get_setting_query import GetSettingQuery
from core.cqrs import QueryHandler, QueryRegistry
from core.result import Result
from shared.messages import SettingResult
from utils import get_logger


@QueryRegistry.register_handler(GetSettingQuery)
class GetSettingQueryHandler(QueryHandler[Result[Any]]):
    """
    Handler xử lý GetSettingQuery để lấy cài đặt.
    """

    def __init__(self):
        """
        Khởi tạo handler
        """
        super().__init__()
        self.logger = get_logger(__name__)
        self.collections = get_collections()

    async def execute(self, query: GetSettingQuery) -> Result[Any]:
        result = await self.collections.settings.find_one({})
        if not result:
            return Result.failure(
                code=SettingResult.NOT_FOUND.code,
                message=SettingResult.NOT_FOUND.message,
            )

        return Result.success(
            message=SettingResult.FETCHED.message,
            code=SettingResult.FETCHED.code,
            data=result
        )
