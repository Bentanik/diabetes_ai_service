"""
Page Location - Value Object cho vị trí trong trang tài liệu

File này định nghĩa PageLocation để lưu trữ và xử lý thông tin về
vị trí của nội dung trong một trang tài liệu.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from app.database.enums import DocumentType
from app.database.value_objects.bounding_box import BoundingBox


@dataclass
class PageLocation:
    """
    Value Object chứa thông tin về vị trí trong trang tài liệu

    Attributes:
        page (int): Số trang
        bbox (BoundingBox): Tọa độ khung chứa nội dung
        block_index (Optional[int]): Chỉ số của block trong trang
        doc_type (DocumentType): Loại tài liệu
    """

    page: int = 0
    bbox: BoundingBox = field(default_factory=BoundingBox)
    block_index: Optional[int] = None
    doc_type: DocumentType = DocumentType.UPLOAD

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi sang dictionary cho MongoDB"""
        return {
            "page": self.page,
            "bbox": self.bbox.to_dict(),
            "block_index": self.block_index,
            "document_type": self.doc_type,
        }
