"""
Chat Model DTO - Model DTO cho cuộc trò chuyện

File này định nghĩa ChatModelDTO để chuyển đổi dữ liệu
giữa ChatModel và API responses.
"""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field

from app.database.models import ChatModel


class ChatModelDTO(BaseModel):
    """
    DTO cho cuộc trò chuyện

    Attributes:
        id (str): ID của cuộc trò chuyện
        session_id (str): ID của phiên trò chuyện
        user_id (str): ID của người dùng
        content (str): Nội dung của cuộc trò chuyện
        response (str): Phản hồi từ hệ thống
        created_at (datetime): Thời điểm tạo
        updated_at (datetime): Thời điểm cập nhật cuối
    """

    id: str = Field(..., description="ID của cuộc trò chuyện")
    session_id: str = Field(..., description="ID của phiên trò chuyện")
    user_id: str = Field(..., description="ID của người dùng")
    content: str = Field(..., description="Nội dung của cuộc trò chuyện")
    response: str = Field(..., description="Phản hồi từ hệ thống")
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @classmethod
    def from_model(cls, model: ChatModel) -> "ChatModelDTO":
        """Tạo DTO từ model"""
        return cls(
            id=model.id,
            session_id=model.session_id,
            user_id=model.user_id,
            content=model.content,
            response=model.response,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )
