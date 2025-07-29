"""
Update Document Command - Command cập nhật thông tin tài liệu

File này định nghĩa UpdateDocumentCommand để cập nhật thông tin
của một tài liệu trong hệ thống.
"""

from dataclasses import dataclass
from typing import Optional
from core.cqrs import Command


@dataclass
class UpdateDocumentCommand(Command):
    """
    Command cập nhật thông tin tài liệu

    Attributes:
        id (str): ID của tài liệu cần cập nhật
        title (Optional[str]): Tiêu đề mới của tài liệu
        description (Optional[str]): Mô tả mới của tài liệu
        priority_diabetes (Optional[float]): Độ ưu tiên liên quan đến tiểu đường
    """

    id: str
    title: Optional[str] = None
    description: Optional[str] = None
    priority_diabetes: Optional[float] = None
