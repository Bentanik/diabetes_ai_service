"""
Document Model DTO - Model DTO cho tài liệu

File này định nghĩa DocumentModelDTO để chuyển đổi dữ liệu
giữa DocumentModel và API responses.
"""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field

from app.database.models import DocumentModel
from app.dto.enums import DocumentType
from app.dto.value_objects import DocumentFileDTO


class DocumentModelDTO(BaseModel):
    """
    DTO cho tài liệu

    Attributes:
        id (str): ID của tài liệu
        knowledge_id (str): ID của cơ sở tri thức chứa tài liệu
        title (str): Tiêu đề của tài liệu
        description (str): Mô tả về tài liệu
        file (DocumentFileDTO): Thông tin về file
        type (DocumentType): Loại tài liệu
        priority_diabetes (float): Độ ưu tiên về tiểu đường (0.0-1.0)
        created_at (datetime): Thời điểm tạo
        updated_at (datetime): Thời điểm cập nhật cuối
    """
    id: str = Field(..., description="ID của tài liệu")
    knowledge_id: str = Field(..., description="ID của cơ sở tri thức")
    title: str = Field(..., min_length=1, description="Tiêu đề tài liệu")
    description: str = Field("", description="Mô tả về tài liệu")
    file: DocumentFileDTO
    type: DocumentType
    priority_diabetes: float = Field(0.0, ge=0.0, le=1.0, description="Độ ưu tiên")
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @classmethod
    def from_model(cls, model: DocumentModel) -> "DocumentModelDTO":
        """Tạo DTO từ model"""
        return cls(
            id=str(model.id),
            knowledge_id=model.knowledge_id,
            title=model.title,
            description=model.description,
            file=DocumentFileDTO.from_value_object(model.file),
            type=DocumentType(model.type.value),
            priority_diabetes=model.priority_diabetes,
            created_at=model.created_at,
            updated_at=model.updated_at
        ) 