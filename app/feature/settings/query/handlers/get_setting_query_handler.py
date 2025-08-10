from typing import Any

from bson import ObjectId
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
        result = await self.collections.settings.find_one(
            {},
        )
        if not result:
            return Result.failure(
                code=SettingResult.NOT_FOUND.code,
                message=SettingResult.NOT_FOUND.message,
            )

        if not result["list_knowledge_id"]:
            return Result.success(
                message=SettingResult.FETCHED.message,
                code=SettingResult.FETCHED.code,
                data={
                    "number_of_passages": result["number_of_passages"],
                    "search_accuracy": result["search_accuracy"],
                    "knowledges": [],
                },
            )

        knowledges = await self.collections.knowledges.find(
            {
                "_id": {
                    "$in": [
                        ObjectId(knowledge_id)
                        for knowledge_id in result["list_knowledge_id"]
                    ]
                }
            },
        ).to_list(length=None)

        return Result.success(
            message=SettingResult.FETCHED.message,
            code=SettingResult.FETCHED.code,
            data={
                "number_of_passages": result["number_of_passages"],
                "search_accuracy": result["search_accuracy"],
                "knowledges": knowledges,
            },
        )
