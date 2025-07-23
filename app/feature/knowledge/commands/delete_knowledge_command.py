from dataclasses import dataclass
from typing import Optional
from core.cqrs.base import Command


@dataclass
class DeleteKnowledgeCommand(Command):
    """
    Command để xóa cơ sở tri thức

    Attributes:
        id: ID của cơ sở tri thức
    """

    id: str

    def __post_init__(self):
        if not self.id:
            raise ValueError("ID của cơ sở tri thức không được để trống")
