"""
Update Knowledge Command - Command để cập nhật cơ sở tri thức

File này định nghĩa UpdateKnowledgeCommand cho phép cập nhật
một hoặc nhiều trường của cơ sở tri thức hiện có.
"""

from dataclasses import dataclass
from typing import Optional
from core.cqrs.base import Command


@dataclass
class UpdateKnowledgeCommand(Command):
    """
    Command để cập nhật cơ sở tri thức
    
    Attributes:
        id (str): ID của cơ sở tri thức cần cập nhật (bắt buộc)
        name (Optional[str]): Tên mới của cơ sở tri thức (optional)
        description (Optional[str]): Mô tả mới về cơ sở tri thức (optional)
        select_training (Optional[bool]): Trạng thái chọn để huấn luyện (optional)
    """

    id: str
    name: Optional[str] = None
    description: Optional[str] = None
    select_training: Optional[bool] = None

    def __post_init__(self):
        """
        Thực hiện validation cơ bản sau khi khởi tạo

        Raises:
            ValueError: Khi ID trống hoặc không hợp lệ
        """
        # Kiểm tra ID không được trống
        if not self.id:
            raise ValueError("ID của cơ sở tri thức không được để trống")
