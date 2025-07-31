"""
Chat Model - Module quản lý các cuộc trò chuyện

File này định nghĩa ChatModel để lưu trữ thông tin về các cuộc trò chuyện
trong hệ thống.
"""

from typing import Dict, Any

from app.database.models import BaseModel


class ChatModel(BaseModel):
    """
    Model cho Chat (Cuộc trò chuyện)

    Attributes:
        Thông tin cơ bản:
            session_id (str): ID của phiên trò chuyện
            user_id (str): ID của người dùng
            content (str): Nội dung của cuộc trò chuyện
            response (str): Phản hồi từ hệ thống
    """

    def __init__(self, session_id: str, user_id: str, content: str, response: str, **kwargs):
        super().__init__(**kwargs)
        self.session_id = session_id
        self.user_id = user_id
        self.content = content
        self.response = response

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatModel":
        """Tạo instance từ MongoDB dictionary"""
        if data is None:
            return None

        # Tạo copy để không modify original data
        data = dict(data)

        # Thông tin cơ bản
        session_id = str(data.pop("session_id", ""))
        user_id = str(data.pop("user_id", ""))
        content = data.pop("content", "")
        response = data.pop("response", "")

        return cls(session_id=session_id, user_id=user_id, content=content, response=response, **data)
