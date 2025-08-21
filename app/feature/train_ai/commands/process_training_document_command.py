from dataclasses import dataclass
from core.cqrs import Command


@dataclass
class ProcessTrainingDocumentCommand(Command):
    """
    Xử lý tài liệu để huấn luyện AI.
    Attributes:
        document_job_id: ID của job tài liệu
    """

    document_job_id: str

    def __post_init__(self):
        if not self.document_job_id:
            raise ValueError("ID của job tài liệu không được để trống")
