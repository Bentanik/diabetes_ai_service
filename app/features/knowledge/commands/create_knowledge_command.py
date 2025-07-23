from dataclasses import dataclass
from core.cqrs.base import Command


@dataclass
class CreateKnowledgeCommand(Command):
    name: str
    description: str

    def __post_init__(self):
        # Validation
        if not self.name or not self.name.strip():
            raise ValueError("Tên thư mục không được để trống")
        if not self.description or not self.description.strip():
            raise ValueError("Mô tả không được để trống")

        self.name = self.name.strip()
        self.description = self.description.strip()
