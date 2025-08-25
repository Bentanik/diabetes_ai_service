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
        document_type (DocumentType): Loại tài liệu
        file (DocumentFileDTO): Thông tin về file
        file_hash (str): Hash của file
        priority_diabetes (float): Độ ưu tiên về tiểu đường (0.0-1.0)
        created_at (datetime): Thời điểm tạo
        updated_at (datetime): Thời điểm cập nhật cuối
    """

    id: str = Field(..., description="ID của tài liệu")
    knowledge_id: str
    title: str
    description: Optional[str] = ""
    document_type: DocumentType = DocumentType.UPLOADED
    priority_diabetes: float = 0.0,
    file: Optional[DocumentFileDTO] = None,
    file_hash: Optional[str] = None,
    is_active: Optional[bool] = True,
    created_at: Optional[datetime] = None,
    updated_at: Optional[datetime] = None

    @classmethod
    def from_model(cls, model: DocumentModel) -> "DocumentModelDTO":
        """Tạo DTO từ model"""
        return cls(
            id=model.id,
            knowledge_id=model.knowledge_id,
            title=model.title,
            description=model.description,
            document_type=model.document_type,
            priority_diabetes=model.priority_diabetes,
            file=DocumentFileDTO.from_value_object(model.file),
            file_hash=model.file_hash,
            is_active=model.is_active,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )
