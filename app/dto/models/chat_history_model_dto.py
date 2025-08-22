"""
Chat History Model DTO - Model DTO cho lịch sử cuộc trò chuyện

File này định nghĩa ChatHistoryModelDTO để chuyển đổi dữ liệu
giữa ChatHistoryModel và API responses.
"""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field

from app.database.models import ChatHistoryModel
from app.dto.enums.chat_role_type import ChatRoleType


class ChatHistoryModelDTO(BaseModel):
    """
    DTO cho lịch sử cuộc trò chuyện

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
    role: ChatRoleType = Field(..., description="Vai trò của người dùng")
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @classmethod
    def from_model(cls, model: ChatHistoryModel) -> "ChatHistoryModelDTO":
        """Tạo DTO từ model"""

        if model.role == ChatRoleType.USER:
            role = ChatRoleType.USER
        else:
            role = ChatRoleType.AI

        return cls(
            id=model.id,
            session_id=model.session_id,
            user_id=model.user_id,
            content=model.content,
            role=role,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )
