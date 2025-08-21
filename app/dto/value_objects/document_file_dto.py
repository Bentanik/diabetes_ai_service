"""
Document File DTO - Value Object DTO cho thông tin file

File này định nghĩa DocumentFileDTO để chuyển đổi dữ liệu
giữa DocumentFile value object và API responses.
"""

from typing import Optional
from pydantic import BaseModel, Field

from app.database.value_objects import DocumentFile


class DocumentFileDTO(BaseModel):
    """
    DTO cho thông tin file của tài liệu

    Attributes:
        path (str): Đường dẫn đến file trong storage
        size_bytes (int): Kích thước file tính bằng bytes
        hash (Optional[str]): Hash của file để kiểm tra trùng lặp
    """

    path: str = Field("", description="Đường dẫn đến file")
    size_bytes: int = Field(0, ge=0, description="Kích thước file (bytes)")
    name: Optional[str] = Field(None, description="Tên file")
    type: Optional[str] = Field(None, description="Loại file")

    @classmethod
    def from_value_object(cls, value_object: DocumentFile) -> "DocumentFileDTO":
        """Tạo DTO từ value object"""
        return cls(
            path=value_object.path,
            size_bytes=value_object.size_bytes,
            name=value_object.name,
            type=value_object.type,
        )

    def to_value_object(self) -> DocumentFile:
        """Chuyển đổi DTO thành value object"""
        return DocumentFile(
            path=self.path,
            size_bytes=self.size_bytes,
            name=self.name,
            type=self.type,
        )
