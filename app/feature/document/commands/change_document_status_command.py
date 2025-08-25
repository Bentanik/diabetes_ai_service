from dataclasses import dataclass
from core.cqrs import Command


@dataclass
class ChangeDocumentChunkStatusCommand(Command):
    """
    Thay đổi trạng thái của tài liệu
    Attributes:
        document_chunk_ids: ID của tài liệu
    """

    document_chunk_ids: list[str]
