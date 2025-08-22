"""
Chat History Model - Module quản lý các cuộc trò chuyện

File này định nghĩa ChatHistoryModel để lưu trữ thông tin về các cuộc trò chuyện
trong hệ thống.
"""

from typing import Dict, Any

from app.database.models import BaseModel
from app.database.enums import ChatRoleType


class ChatHistoryModel(BaseModel):
    """
    Model cho Chat History (Lịch sử cuộc trò chuyện)

    Attributes:
        Thông tin cơ bản:
            session_id (str): ID của phiên trò chuyện
            user_id (str): ID của người dùng
            content (str): Nội dung của cuộc trò chuyện
            role (str): Vai trò của người dùng
    """

    def __init__(
        self, session_id: str, user_id: str, content: str, role: ChatRoleType, **kwargs
    ):
        super().__init__(**kwargs)
        self.session_id = session_id
        self.user_id = user_id
        self.content = content
        self.role = role

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatHistoryModel":
        """Tạo instance từ MongoDB dictionary"""
        if data is None:
            return None

        # Tạo copy để không modify original data
        data = dict(data)

        # Thông tin cơ bản
        session_id = str(data.pop("session_id", ""))
        user_id = str(data.pop("user_id", ""))
        content = data.pop("content", "")
        role = data.pop("role", "")


        return cls(
            session_id=session_id, user_id=user_id, content=content, role=role, **data
        )
