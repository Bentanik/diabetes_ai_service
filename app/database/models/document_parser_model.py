"""
Document Parser Model - Module xử lý và phân tích tài liệu

File này định nghĩa DocumentParserModel để lưu trữ kết quả phân tích
và trích xuất nội dung từ tài liệu.
"""

from typing import Dict, Union

from app.database.models import BaseModel
from app.database.value_objects import PageLocation

DocumentParserDict = Dict[str, Union[str, bool, dict, None]]


class DocumentParserModel(BaseModel):
    """
    Model quản lý kết quả phân tích tài liệu

    Attributes:
        document_id (str): ID của tài liệu gốc
        content (str): Nội dung được trích xuất
        location (PageLocation): Vị trí của nội dung trong tài liệu
        is_active (bool): Trạng thái hoạt động của bản ghi
    """

    def __init__(
        self,
        document_id: str,
        content: str,
        location: PageLocation,
        is_active: bool = True,
        **kwargs
    ):
        """Khởi tạo một bản ghi phân tích tài liệu"""
        super().__init__(**kwargs)
        self.document_id = document_id
        self.content = content
        self.location = location
        self.is_active = is_active

    def to_dict(self) -> DocumentParserDict:
        """Chuyển đổi sang dictionary"""
        result = super().to_dict()
        result.update(
            {
                "document_id": self.document_id,
                "content": self.content,
                "metadata": self.location.to_dict(),
                "is_active": self.is_active,
            }
        )
        return result

    @classmethod
    def from_dict(cls, data: DocumentParserDict) -> "DocumentParserModel":
        """Tạo instance từ dictionary"""
        data = dict(data)
        document_id = data.pop("document_id", None)
        content = data.pop("content", "")
        location = PageLocation.from_dict(data.pop("metadata", {}))
        is_active = data.pop("is_active", True)
        
        return cls(
            document_id=document_id,
            content=content,
            location=location,
            is_active=is_active,
            **data,
        )
