"""
Chat Session Model DTO - Model DTO cho phiên trò chuyện

File này định nghĩa ChatSessionModelDTO để chuyển đổi dữ liệu
giữa ChatSessionModel và API responses.
"""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field

from app.database.models import ChatSessionModel


class ChatSessionModelDTO(BaseModel):
    """
    DTO cho phiên trò chuyện

    Attributes:
        id (str): ID của phiên trò chuyện
        user_id (str): ID của người dùng
        title (str): Tiêu đề của phiên trò chuyện
        external_knowledge (bool): Có sử dụng tri thức bên ngoài hay không
        created_at (datetime): Thời điểm tạo
        updated_at (datetime): Thời điểm cập nhật cuối
    """

    id: str = Field(..., description="ID của cuộc trò chuyện")
    user_id: str = Field(..., description="ID của người dùng")
    title: str = Field(..., description="Tiêu đề của phiên trò chuyện")
    external_knowledge: bool = Field(..., description="Có sử dụng tri thức bên ngoài hay không")
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @classmethod
    def from_model(cls, model: ChatSessionModel) -> "ChatSessionModelDTO":
        """Tạo DTO từ model"""
        return cls(
            id=model.id,
            user_id=model.user_id,
            title=model.title,
            external_knowledge=model.external_knowledge,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )
