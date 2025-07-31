"""
Delete Knowledge Command - Command để xóa cơ sở tri thức

File này định nghĩa DeleteKnowledgeCommand, cần ID của cơ sở tri thức
để thực hiện thao tác xóa.
"""

from dataclasses import dataclass
from core.cqrs import Command


@dataclass
class DeleteKnowledgeCommand(Command):
    """
    Command để xóa cơ sở tri thức
    
    Attributes:
        id (str): ID của cơ sở tri thức cần xóa (bắt buộc)
    """

    id: str

    def __post_init__(self):
        """
        Thực hiện validation cơ bản sau khi khởi tạo

        Raises:
            ValueError: Khi ID trống hoặc không hợp lệ
        """
        # Kiểm tra ID không được trống
        if not self.id:
            raise ValueError("ID của cơ sở tri thức không được để trống")
