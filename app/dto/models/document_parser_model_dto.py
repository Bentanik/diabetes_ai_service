"""
Document Parser Model DTO - Model DTO cho kết quả phân tích tài liệu

File này định nghĩa DocumentParserModelDTO để chuyển đổi dữ liệu
giữa DocumentParserModel và API responses.
"""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field

from app.database.models import DocumentParserModel
from app.dto.value_objects import PageLocationDTO


class DocumentParserModelDTO(BaseModel):
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
    content: str = Field(..., description="Nội dung được trích xuất")
    location: PageLocationDTO
    is_active: bool = Field(True, description="Trạng thái hoạt động")
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @classmethod
    def from_model(cls, model: DocumentParserModel) -> "DocumentParserModelDTO":
        """Tạo DTO từ model"""
        return cls(
            id=str(model.id),
            document_id=model.document_id,
            content=model.content,
            location=PageLocationDTO.from_value_object(model.location),
            is_active=model.is_active,
            created_at=model.created_at,
            updated_at=model.updated_at
        ) 