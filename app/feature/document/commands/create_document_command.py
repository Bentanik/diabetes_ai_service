from dataclasses import dataclass
from fastapi import UploadFile
from core.cqrs import Command


@dataclass
class CreateDocumentCommand(Command):
    """
    Tạo tài liệu mới
    Attributes:
        file: Tệp tài liệu
        knowledge_id: ID của cơ sở tri thức
    """

    file: UploadFile
    knowledge_id: str
    title: str
    description: str

    def __post_init__(self):
        if not self.title:
            raise ValueError("Tên tài liệu không được để trống")

        if not self.description:
            raise ValueError("Mô tả tài liệu không được để trống")

        if not self.knowledge_id:
            raise ValueError("ID của cơ sở tri thức không được để trống")

        self.title = self.title.strip()
        self.description = self.description.strip()
