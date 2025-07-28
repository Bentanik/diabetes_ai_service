"""
Processing Status DTO - Value Object DTO cho trạng thái xử lý

File này định nghĩa ProcessingStatusDTO để chuyển đổi dữ liệu
giữa ProcessingStatus value object và API responses.
"""

from pydantic import BaseModel, Field

from app.database.value_objects import ProcessingStatus
from app.dto.enums import DocumentJobStatus


class ProcessingStatusDTO(BaseModel):
    """
    DTO cho trạng thái xử lý

    Attributes:
        status (DocumentJobStatus): Trạng thái hiện tại của công việc
        progress (float): Tiến độ hoàn thành (0.0 - 1.0)
        message (str): Thông báo về tiến độ hoặc lỗi
    """
    status: DocumentJobStatus = Field(DocumentJobStatus.PENDING, description="Trạng thái")
    progress: float = Field(0.0, ge=0.0, le=1.0, description="Tiến độ (0.0-1.0)")
    message: str = Field("", description="Thông báo")

    @classmethod
    def from_value_object(cls, value_object: ProcessingStatus) -> "ProcessingStatusDTO":
        """Tạo DTO từ value object"""
        return cls(
            status=DocumentJobStatus(value_object.status.value),
            progress=value_object.progress,
            message=value_object.message
        )

    def to_value_object(self) -> ProcessingStatus:
        """Chuyển đổi DTO thành value object"""
        return ProcessingStatus(
            status=self.status,
            progress=self.progress,
            message=self.message
        ) 