"""
Chat Query - Truy vấn lấy thông tin chat

File này định nghĩa ChatQuery để lấy thông tin chi tiết
của một chat từ database dựa trên ID.
"""

from dataclasses import dataclass
from core.cqrs import Query


@dataclass
class ChatQuery(Query):
    """
    Query lấy thông tin chi tiết một chat
    """

    session_id: str
    user_id: str
    content: str
