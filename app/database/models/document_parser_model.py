"""
Document Parser Model - Module xử lý và phân tích tài liệu

File này định nghĩa DocumentParserModel để lưu trữ kết quả phân tích
và trích xuất nội dung từ tài liệu.
"""

from typing import Dict, Any
from app.database.models import BaseModel
from app.database.value_objects import PageLocation
from app.database.value_objects.bounding_box import BoundingBox
from app.database.enums import DocumentType


class DocumentParserModel(BaseModel):
    """
    Model quản lý kết quả phân tích tài liệu

    Attributes:
        document_id (str): ID của tài liệu gốc
        knowledge_id (str): ID của cơ sở tri thức liên quan
        content (str): Nội dung được trích xuất
        location (PageLocation): Vị trí của nội dung trong tài liệu
        is_active (bool): Trạng thái hoạt động của bản ghi
    """

    def __init__(
        self,
        document_id: str,
        knowledge_id: str,
        content: str,
        location: PageLocation,
        is_active: bool = True,
        **kwargs
    ):
        """Khởi tạo một bản ghi phân tích tài liệu"""
        super().__init__(**kwargs)
        self.document_id = document_id
        self.knowledge_id = knowledge_id
        self.content = content
        self.location = location
        self.is_active = is_active

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentParserModel":
        """Tạo instance từ MongoDB dictionary"""
        if data is None:
            return None

        # Tạo copy để không modify original data
        data = dict(data)

        document_id = data.pop("document_id", "")
        knowledge_id = data.pop("knowledge_id", "")
        content = data.pop("content", "")
        is_active = data.pop("is_active", True)

        # Tạo PageLocation từ metadata hoặc các field riêng lẻ
        bbox_data = data.pop("bbox", {})
        bbox = BoundingBox(
            x0=bbox_data.get("x0", 0.0),
            y0=bbox_data.get("y0", 0.0),
            x1=bbox_data.get("x1", 0.0),
            y1=bbox_data.get("y1", 0.0),
        )

        location = PageLocation(
            page=data.pop("page", 0),
            bbox=bbox,
            block_index=data.pop("block_index", None),
            doc_type=data.pop("document_type", DocumentType.UPLOAD),
        )

        return cls(
            document_id=document_id,
            knowledge_id=knowledge_id,
            content=content,
            location=location,
            is_active=is_active,
            **data
        )
