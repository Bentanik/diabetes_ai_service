"""
Knowledge Model DTO - Model DTO cho cơ sở tri thức

File này định nghĩa KnowledgeModelDTO để chuyển đổi dữ liệu
giữa KnowledgeModel và API responses.
"""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field

from app.database.models import KnowledgeModel
from app.dto.value_objects import KnowledgeStatsDTO


class KnowledgeModelDTO(BaseModel):
    """
    DTO cho cơ sở tri thức

    Attributes:
        id (str): ID của cơ sở tri thức
        name (str): Tên của cơ sở tri thức
        description (str): Mô tả về cơ sở tri thức
        select_training (bool): Đánh dấu có được chọn để huấn luyện
        stats (KnowledgeStatsDTO): Thống kê về số lượng và dung lượng tài liệu
        created_at (datetime): Thời điểm tạo
        updated_at (datetime): Thời điểm cập nhật cuối
    """

    id: str = Field(..., description="ID của cơ sở tri thức")
    name: str = Field(..., min_length=1, description="Tên cơ sở tri thức")
    description: str = Field("", description="Mô tả về cơ sở tri thức")
    select_training: bool = Field(False, description="Chọn để huấn luyện")
    stats: KnowledgeStatsDTO
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @classmethod
    def from_model(cls, model: KnowledgeModel) -> "KnowledgeModelDTO":
        """Tạo DTO từ model"""
        return cls(
            id=str(model.id),
            name=model.name,
            description=model.description,
            select_training=model.select_training,
            stats=KnowledgeStatsDTO.from_value_object(model.stats),
            created_at=model.created_at,
            updated_at=model.updated_at,
        )
