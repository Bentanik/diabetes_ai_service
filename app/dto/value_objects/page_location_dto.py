"""
Page Location DTO - Value Object DTO cho vị trí trong trang

File này định nghĩa PageLocationDTO để chuyển đổi dữ liệu
giữa PageLocation value object và API responses.
"""

from typing import Optional
from pydantic import BaseModel, Field

from app.database.value_objects import PageLocation
from app.dto.enums import DocumentType
from app.dto.value_objects.bounding_box_dto import BoundingBoxDTO


class PageLocationDTO(BaseModel):
    """
    DTO cho vị trí trong trang tài liệu

    Attributes:
        source (str): Nguồn của trang (thường là tên file)
        page (int): Số trang
        bbox (BoundingBoxDTO): Tọa độ khung chứa nội dung
        block_index (Optional[int]): Chỉ số của block trong trang
        doc_type (DocumentType): Loại tài liệu
    """
    source: str = Field("", description="Nguồn của trang")
    page: int = Field(0, ge=0, description="Số trang")
    bbox: BoundingBoxDTO
    block_index: Optional[int] = Field(None, description="Chỉ số block")
    doc_type: DocumentType = Field(DocumentType.UPLOAD, description="Loại tài liệu")

    @classmethod
    def from_value_object(cls, value_object: PageLocation) -> "PageLocationDTO":
        """Tạo DTO từ value object"""
        return cls(
            source=value_object.source,
            page=value_object.page,
            bbox=BoundingBoxDTO.from_value_object(value_object.bbox),
            block_index=value_object.block_index,
            doc_type=DocumentType(value_object.doc_type.value)
        )

    def to_value_object(self) -> PageLocation:
        """Chuyển đổi DTO thành value object"""
        return PageLocation(
            source=self.source,
            page=self.page,
            bbox=self.bbox.to_value_object(),
            block_index=self.block_index,
            doc_type=self.doc_type
        ) 