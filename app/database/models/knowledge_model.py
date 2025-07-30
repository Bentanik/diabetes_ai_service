"""
Knowledge Model - Module quản lý cơ sở tri thức

File này định nghĩa KnowledgeModel để lưu trữ và quản lý thông tin
về các cơ sở tri thức trong hệ thống.
"""

from typing import Any, Dict, Optional

from app.database.models import BaseModel
from app.database.value_objects import KnowledgeStats


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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeModel":
        """Tạo instance từ MongoDB dictionary"""
        if data is None:
            return None

        # Tạo copy để không modify original data
        data = dict(data)

        # Thông tin cơ bản
        name = data.pop("name", "")
        description = data.pop("description", "")
        select_training = data.pop("select_training", False)

        # Tạo KnowledgeStats từ dữ liệu thống kê
        stats = KnowledgeStats(
            document_count=data.pop("document_count", 0),
            total_size_bytes=data.pop("total_size_bytes", 0),
        )

        return cls(
            name=name,
            description=description,
            select_training=select_training,
            stats=stats,
            **data  # Các field còn lại như _id, created_at, updated_at
        )
