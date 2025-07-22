from typing import Dict, Union, Optional
from bson import ObjectId
from datetime import datetime

from app.database.models.base_model import BaseModel

KnowledgeDict = Dict[str, Union[str, int, bool, datetime, ObjectId, None]]


class KnowledgeModel(BaseModel):
    """Model cho Knowledge (Cơ sở tri thức)"""

    def __init__(
        self,
        name: str,
        description: Optional[str] = "",
        document_count: int = 0,
        total_size_bytes: int = 0,
        select_training: bool = False,
        **kwargs
    ):
        """
        Args:
            name (str): Tên của cơ sở tri thức.
            description (Optional[str]): Mô tả về cơ sở tri thức.
            document_count (int): Số lượng tài liệu trong cơ sở tri thức.
            total_size_bytes (int): Tổng dung lượng (bytes) của các tài liệu.
            select_training (bool): Đánh dấu có chọn để huấn luyện hay không.
        """
        super().__init__(**kwargs)
        self.name = name  # Tên của cơ sở tri thức
        self.description = description  # Mô tả về cơ sở tri thức
        self.document_count = document_count  # Số lượng tài liệu
        self.total_size_bytes = total_size_bytes  # Tổng dung lượng các tài liệu (bytes)
        self.select_training = select_training  # Có chọn để huấn luyện hay không

    def to_dict(self) -> KnowledgeDict:
        result = super().to_dict()
        result.update(
            {
                "name": self.name,
                "description": self.description,
                "document_count": self.document_count,
                "total_size_bytes": self.total_size_bytes,
                "select_training": self.select_training,
            }
        )
        return result

    @classmethod
    def from_dict(cls, data: KnowledgeDict) -> "KnowledgeModel":
        data = dict(data)
        name = data.pop("name", "")
        description = data.pop("description", "")
        document_count = data.pop("document_count", 0)
        total_size_bytes = data.pop("total_size_bytes", 0)
        select_training = data.pop("select_training", False)
        return cls(
            name=name,
            description=description,
            document_count=document_count,
            total_size_bytes=total_size_bytes,
            select_training=select_training,
            **data
        )
