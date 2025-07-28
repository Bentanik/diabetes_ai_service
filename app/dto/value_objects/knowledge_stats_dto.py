"""
Knowledge Stats DTO - Value Object DTO cho thống kê cơ sở tri thức

File này định nghĩa KnowledgeStatsDTO để chuyển đổi dữ liệu
giữa KnowledgeStats value object và API responses.
"""

from pydantic import BaseModel, Field

from app.database.value_objects import KnowledgeStats


class KnowledgeStatsDTO(BaseModel):
    """
    DTO cho thống kê của cơ sở tri thức

    Attributes:
        document_count (int): Số lượng tài liệu
        total_size_bytes (int): Tổng dung lượng của các tài liệu (bytes)
    """
    document_count: int = Field(0, ge=0, description="Số lượng tài liệu")
    total_size_bytes: int = Field(0, ge=0, description="Tổng dung lượng (bytes)")

    @classmethod
    def from_value_object(cls, value_object: KnowledgeStats) -> "KnowledgeStatsDTO":
        """Tạo DTO từ value object"""
        return cls(
            document_count=value_object.document_count,
            total_size_bytes=value_object.total_size_bytes
        )

    def to_value_object(self) -> KnowledgeStats:
        """Chuyển đổi DTO thành value object"""
        return KnowledgeStats(
            document_count=self.document_count,
            total_size_bytes=self.total_size_bytes
        ) 