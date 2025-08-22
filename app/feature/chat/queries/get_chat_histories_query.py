from dataclasses import dataclass
from core.cqrs import Query
from typing import Optional


@dataclass
class GetChatHistoriesQuery(Query):
    session_id: Optional[str]
    user_id: Optional[str]
