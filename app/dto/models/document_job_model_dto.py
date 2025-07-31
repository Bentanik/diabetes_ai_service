"""
Document Job Model DTO - Model DTO cho công việc xử lý tài liệu

File này định nghĩa DocumentJobModelDTO để chuyển đổi dữ liệu
giữa DocumentJobModel và API responses.
"""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field

from app.database.models import DocumentJobModel
from app.dto.enums import DocumentJobType
from app.dto.value_objects import ProcessingStatusDTO


class DocumentJobModelDTO(BaseModel):
    """
    DTO cho công việc xử lý tài liệu

    Attributes:
        id (str): ID của công việc
        document_id (str): ID của tài liệu cần xử lý
        knowledge_id (str): ID của cơ sở tri thức chứa tài liệu
        title (str): Tiêu đề tài liệu
        description (str): Mô tả tài liệu
        file_path (str): Đường dẫn đến file
        type (DocumentJobType): Loại công việc
        processing (ProcessingStatusDTO): Trạng thái và tiến độ xử lý
        priority_diabetes (float): Độ ưu tiên về tiểu đường
        is_document_delete (bool): Có xóa tài liệu gốc chưa
        created_at (datetime): Thời điểm tạo
        updated_at (datetime): Thời điểm cập nhật cuối
    """
    id: str = Field(..., description="ID của công việc")
    document_id: str = Field(..., description="ID của tài liệu")
    knowledge_id: str = Field(..., description="ID của cơ sở tri thức")
    title: str = Field(..., min_length=1, description="Tiêu đề tài liệu")
    description: str = Field("", description="Mô tả tài liệu")
    file_path: str = Field("", description="Đường dẫn file")
    type: DocumentJobType
    processing: ProcessingStatusDTO
    priority_diabetes: float = Field(0.0, ge=0.0, le=1.0, description="Độ ưu tiên")
    is_document_delete: bool = Field(False, description="Có xóa tài liệu gốc chưa")
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @classmethod
    def from_model(cls, model: DocumentJobModel) -> "DocumentJobModelDTO":
        """Tạo DTO từ model"""
        return cls(
            id=str(model.id),
            document_id=model.document_id,
            knowledge_id=model.knowledge_id,
            title=model.title,
            description=model.description,
            file_path=model.file_path,
            type=DocumentJobType(model.type.value),
            processing=ProcessingStatusDTO.from_value_object(model.processing),
            priority_diabetes=model.priority_diabetes,
            is_document_delete=model.is_document_delete,
            created_at=model.created_at,
            updated_at=model.updated_at
        ) 