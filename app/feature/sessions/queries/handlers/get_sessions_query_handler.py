from core.cqrs import QueryHandler, QueryRegistry
from core.result import Result
from shared.messages import SessionChatMessage
from utils import get_logger
from app.database import get_collections
from app.database.models import ChatSessionModel
from app.dto.models import ChatSessionModelDTO
from typing import List
from ..get_sessions_query import GetSessionsQuery


@QueryRegistry.register_handler(GetSessionsQuery)
class GetSessionsQueryHandler(QueryHandler[Result[List[ChatSessionModelDTO]]]):
    def __init__(self):
        super().__init__()
        self.db = get_collections()
        self.logger = get_logger(__name__)

    async def execute(
        self, query: GetSessionsQuery
    ) -> Result[List[ChatSessionModelDTO]]:
        try:
            sessions = await self.db.chat_sessions.find(
                {"user_id": query.user_id}
            ).to_list(length=100)
            if not sessions:
                return Result.success(
                    message=SessionChatMessage.SESSIONS_FETCHED_SUCCESS.message,
                    code=SessionChatMessage.SESSIONS_FETCHED_SUCCESS.code,
                    data=[],
                )

            sessions = [ChatSessionModel.from_dict(session) for session in sessions]

            session_dtos = [
                ChatSessionModelDTO.from_model(session) for session in sessions
            ]

            return Result.success(
                message=SessionChatMessage.SESSIONS_FETCHED_SUCCESS.message,
                code=SessionChatMessage.SESSIONS_FETCHED_SUCCESS.code,
                data=session_dtos,
            )

        except Exception as e:
            self.logger.error(
                f"Lỗi khi lấy danh sách phiên trò chuyện: {e}", exc_info=True
            )
            return Result.failure(message="Lỗi hệ thống", code="error")
