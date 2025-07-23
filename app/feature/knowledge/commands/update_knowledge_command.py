from dataclasses import dataclass
from typing import Optional
from core.cqrs.base import Command


@dataclass
class UpdateKnowledgeCommand(Command):
    """
    Command để cập nhật cơ sở tri thức

    Attributes:
        id: ID của cơ sở tri thức
        name: Tên cơ sở tri thức
        description: Mô tả chi tiết về cơ sở tri thức
        select_training: Đánh dấu có chọn để huấn luyện hay không
    """

    id: str
    name: Optional[str] = None
    description: Optional[str] = None
    select_training: Optional[bool] = None

    def __post_init__(self):
        if not self.id:
            raise ValueError("ID của cơ sở tri thức không được để trống")
