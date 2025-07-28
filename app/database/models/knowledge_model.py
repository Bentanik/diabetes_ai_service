"""
Knowledge Model - Module quản lý cơ sở tri thức

File này định nghĩa KnowledgeModel để lưu trữ và quản lý thông tin
về các cơ sở tri thức trong hệ thống.
"""

from typing import Dict, Union, Optional
from datetime import datetime
from bson import ObjectId

from app.database.models import BaseModel
from app.database.value_objects import KnowledgeStats

KnowledgeDict = Dict[str, Union[str, int, bool, datetime, ObjectId, None]]


class KnowledgeModel(BaseModel):
    """
    Model quản lý cơ sở tri thức

    Attributes:
        Thông tin cơ bản:
            name (str): Tên của cơ sở tri thức
            description (str): Mô tả về cơ sở tri thức
            select_training (bool): Đánh dấu có được chọn để huấn luyện hay không

        Thông tin thống kê:
            stats (KnowledgeStats): Thống kê về số lượng và dung lượng tài liệu
    """

    def __init__(
        self,
        name: str,
        description: Optional[str] = "",
        select_training: bool = False,
        stats: Optional[KnowledgeStats] = None,
        **kwargs
    ):
        """Khởi tạo một cơ sở tri thức mới"""
        super().__init__(**kwargs)
        # Thông tin cơ bản
        self.name = name
        self.description = description
        self.select_training = select_training

        # Thông tin thống kê
        self.stats = stats or KnowledgeStats()

    def to_dict(self) -> KnowledgeDict:
        """Chuyển đổi sang dictionary"""
        result = super().to_dict()
        result.update(
            {
                # Thông tin cơ bản
                "name": self.name,
                "description": self.description,
                "select_training": self.select_training,
                # Thông tin thống kê
                **self.stats.to_dict(),
            }
        )
        return result

    @classmethod
    def from_dict(cls, data: KnowledgeDict) -> "KnowledgeModel":
        """Tạo instance từ dictionary"""
        data = dict(data)

        # Thông tin cơ bản
        name = data.pop("name", "")
        description = data.pop("description", "")
        select_training = data.pop("select_training", False)

        # Tạo KnowledgeStats từ dữ liệu thống kê
        stats = KnowledgeStats.from_dict(data)

        # Xóa các trường thống kê khỏi data để tránh trùng lặp
        data.pop("document_count", None)
        data.pop("total_size_bytes", None)

        return cls(
            name=name,
            description=description,
            select_training=select_training,
            stats=stats,
            **data,
        )
