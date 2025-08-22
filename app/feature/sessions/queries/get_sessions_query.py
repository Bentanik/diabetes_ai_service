from dataclasses import dataclass
from core.cqrs import Query


@dataclass
class GetSessionsQuery(Query):
    """
    Query lấy danh sách phiên trò chuyện
    """

    user_id: str
