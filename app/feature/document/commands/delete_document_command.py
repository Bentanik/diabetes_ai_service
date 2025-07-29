"""
Delete Document Command - Command xóa tài liệu

File này định nghĩa DeleteDocumentCommand để xóa một tài liệu
khỏi hệ thống.
"""

from dataclasses import dataclass
from core.cqrs import Command


@dataclass
class DeleteDocumentCommand(Command):
    """
    Command xóa tài liệu

    Attributes:
        id (str): ID của tài liệu cần xóa
    """

    id: str
