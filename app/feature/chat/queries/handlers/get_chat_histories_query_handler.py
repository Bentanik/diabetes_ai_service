from core.cqrs import QueryHandler, QueryRegistry
from core.result import Result
from shared.messages import ChatMessage
from utils import get_logger
from app.database import get_collections
from app.database.models import ChatHistoryModel
from app.dto.models import ChatHistoryModelDTO
from typing import List
from ..get_chat_histories_query import GetChatHistoriesQuery


@QueryRegistry.register_handler(GetChatHistoriesQuery)
class GetChatHistoriesQueryHandler(QueryHandler[Result[List[ChatHistoryModelDTO]]]):
    def __init__(self):
        super().__init__()
        self.db = get_collections()
        self.logger = get_logger(__name__)

    async def execute(
        self, query: GetChatHistoriesQuery
    ) -> Result[List[ChatHistoryModelDTO]]:
        try:
            chat_histories = []

            if query.user_id != "admin":
                chat_histories = (
                    await self.db.chat_histories.find({"session_id": query.session_id})
                    .sort("created_at", -1)
                    .to_list(length=None)
                )
            else:
                chat_histories = (
                    await self.db.chat_histories.find({"user_id": query.user_id})
                    .sort("created_at", -1)
                    .to_list(length=None)
                )

            chat_histories = [
                ChatHistoryModelDTO.from_model(ChatHistoryModel.from_dict(chat_history))
                for chat_history in chat_histories
            ]

            chat_histories.reverse()

            if not chat_histories:
                return Result.success(
                    message=ChatMessage.CHAT_HISTORIES_FETCHED_SUCCESS.message,
                    code=ChatMessage.CHAT_HISTORIES_FETCHED_SUCCESS.code,
                    data=[],
                )

            return Result.success(
                message=ChatMessage.CHAT_HISTORIES_FETCHED_SUCCESS.message,
                code=ChatMessage.CHAT_HISTORIES_FETCHED_SUCCESS.code,
                data=chat_histories,
            )

        except Exception as e:
            self.logger.error(
                f"Lỗi khi lấy danh sách phiên trò chuyện: {e}", exc_info=True
            )
            return Result.failure(message="Lỗi hệ thống", code="error")
