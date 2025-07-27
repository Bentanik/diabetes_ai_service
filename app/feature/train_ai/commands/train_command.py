from dataclasses import dataclass
from core.cqrs import Command


@dataclass
class TrainCommand(Command):
    """
    Huấn luyện mô hình RAG
    Attributes:
        document_id: ID của tài liệu
    """

    document_id: str

    def __post_init__(self):
        if not self.document_id:
            raise ValueError("ID của tài liệu không được để trống")

        self.document_id = self.document_id.strip()
