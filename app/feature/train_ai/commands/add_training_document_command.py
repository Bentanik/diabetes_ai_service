from dataclasses import dataclass
from core.cqrs import Command


@dataclass
class AddTrainingDocumentCommand(Command):
    """
    Thêm tài liệu vào vector database để huấn luyện AI.
    Attributes:
        document_id: ID của tài liệu
    """

    document_id: str

    def __post_init__(self):
        if not self.document_id:
            raise ValueError("ID của tài liệu không được để trống")
