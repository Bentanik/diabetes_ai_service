from dataclasses import dataclass
from core.cqrs import Command


@dataclass
class ProcessDocumentUploadCommand(Command):
    """
    Tạo tài liệu mới
    Attributes:
        file: Tệp tài liệu
        knowledge_id: ID của cơ sở tri thức
    """

    file_path: str
    knowledge_id: str
    document_id: str
    title: str
    description: str

    def __post_init__(self):
        if not self.file_path:
            raise ValueError("Đường dẫn tệp không được để trống")

        if not self.knowledge_id:
            raise ValueError("ID của cơ sở tri thức không được để trống")

        if not self.title:
            raise ValueError("Tên tài liệu không được để trống")

        if not self.description:
            raise ValueError("Mô tả tài liệu không được để trống")

        if not self.document_id:
            raise ValueError("ID của tài liệu không được để trống")

        self.title = self.title.strip()
        self.description = self.description.strip()
