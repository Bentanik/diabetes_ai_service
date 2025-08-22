"""
Create Chat Command - Lệnh tạo cuộc trò chuyện

File này định nghĩa CreateChatCommand để tạo cuộc trò chuyện với AI.
"""

from dataclasses import dataclass
from core.cqrs import Command
from typing import Optional


@dataclass
class CreateChatCommand(Command):
    """
    Command trò chuyện với AI

    Attributes:
        session_id (str): ID của phiên trò chuyện
        content (str): Nội dung của cuộc trò chuyện
        user_id (str): ID của người dùng
        use_external_knowledge (bool): Sử dụng tri thức ngoài
    """

    content: str
    user_id: str
    session_id: Optional[str] = None
    use_external_knowledge: Optional[bool] = False
