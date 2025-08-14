"""
Chat Session Model - Module chứa các phiên trò chuyện

File này định nghĩa ChatSessionModel để lưu trữ thông tin về các phiên trò chuyện
trong hệ thống.
"""

from typing import Dict, Any

from app.database.models import BaseModel


class ChatSessionModel(BaseModel):
    """
    Model cho Chat Session (Phiên trò chuyện)

    Attributes:
        Thông tin cơ bản:
            user_id (str): ID của người dùng
            title (str): Tiêu đề của phiên trò chuyện
            external_knowledge (bool): Có sử dụng tri thức bên ngoài hay không
    """

    def __init__(self, user_id: str, title: str, external_knowledge, **kwargs):
        super().__init__(**kwargs)
        self.user_id = user_id
        self.title = title
        self.external_knowledge = external_knowledge

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatSessionModel":
        """Tạo instance từ MongoDB dictionary"""
        if data is None:
            return None

        # Tạo copy để không modify original data
        data = dict(data)

        # Thông tin cơ bản
        user_id = str(data.pop("user_id", ""))
        title = data.pop("title", "")
        external_knowledge = data.pop("external_knowledge", False)

        return cls(user_id=user_id, title=title, external_knowledge=external_knowledge, **data)
