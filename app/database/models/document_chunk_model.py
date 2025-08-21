"""
Document Chunk Model - Module xử lý và phân tích tài liệu

File này định nghĩa DocumentChunkModel để lưu trữ kết quả phân tích
và trích xuất nội dung từ tài liệu.
"""

from typing import Dict, Any
from app.database.models import BaseModel

class DocumentChunkModel(BaseModel):
    """
    Model quản lý kết quả phân tích tài liệu

    Attributes:
        document_id (str): ID của tài liệu gốc
        knowledge_id (str): ID của cơ sở tri thức liên quan
        content (str): Nội dung được trích xuất
        diabetes_score (float): Điểm số của tài liệu
        is_active (bool): Trạng thái hoạt động của bản ghi
    """

    def __init__(
        self,
        document_id: str,
        knowledge_id: str,
        content: str,
        diabetes_score: float = 0.0,
        is_active: bool = True,
        **kwargs
    ):
        """Khởi tạo một bản ghi phân tích tài liệu"""
        super().__init__(**kwargs)
        self.document_id = document_id
        self.knowledge_id = knowledge_id
        self.content = content
        self.diabetes_score = diabetes_score
        self.is_active = is_active

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentChunkModel":
        if data is None:
            return None

        # Tạo copy để không modify original data
        data = dict(data)

        document_id = data.pop("document_id", "")
        knowledge_id = data.pop("knowledge_id", "")
        content = data.pop("content", "")
        diabetes_score = data.pop("diabetes_score", 0.0)
        is_active = data.pop("is_active", True)

        return cls(
            document_id=document_id,
            knowledge_id=knowledge_id,
            content=content,
            diabetes_score=diabetes_score,
            is_active=is_active,
            **data
        )
