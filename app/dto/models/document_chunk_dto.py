"""
Document Chunk Model DTO - Model DTO cho kết quả phân tích tài liệu

File này định nghĩa DocumentChunkModelDTO để chuyển đổi dữ liệu
giữa DocumentChunkModel và API responses.
"""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field

from app.database.models import DocumentChunkModel


class DocumentChunkModelDTO(BaseModel):
    """
    DTO cho kết quả phân tích tài liệu

    Attributes:
        id (str): ID của bản ghi
        document_id (str): ID của tài liệu gốc
        content (str): Nội dung được trích xuất
        location (PageLocationDTO): Vị trí của nội dung trong tài liệu
        is_active (bool): Trạng thái hoạt động của bản ghi
        created_at (datetime): Thời điểm tạo
        updated_at (datetime): Thời điểm cập nhật cuối
    """
    id: str = Field(..., description="ID của bản ghi")
    document_id: str = Field(..., description="ID của tài liệu gốc")
    knowledge_id: str = Field(..., description="ID của cơ sở tri thức liên quan")
    content: str = Field(..., description="Nội dung được trích xuất")
    diabetes_score: float = Field(0.0, description="Điểm số của tài liệu")
    is_active: bool = Field(True, description="Trạng thái hoạt động")
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @classmethod
    def from_model(cls, model: DocumentChunkModel) -> "DocumentChunkModelDTO":
        """Tạo DTO từ model"""
        return cls(
            id=str(model.id),
            document_id=model.document_id,
            knowledge_id=model.knowledge_id,
            content=model.content,
            diabetes_score=model.diabetes_score,
            is_active=model.is_active,
            created_at=model.created_at,
            updated_at=model.updated_at
        ) 