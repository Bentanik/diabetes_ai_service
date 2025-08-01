from dataclasses import dataclass
from core.cqrs import Query


@dataclass
class GetChatHistoriesQuery(Query):
    session_id: str
