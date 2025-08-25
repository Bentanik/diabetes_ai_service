from dataclasses import dataclass
from typing import Optional
from core.cqrs import Command


@dataclass
class UpdateDocumentCommand(Command):
    """
    Thay đổi trạng thái của tài liệu
    Attributes:
        document_id: ID của tài liệu
        title: Tiêu đề của tài liệu
        description: Mô tả của tài liệu
    """

    document_id: str
    title: Optional[str] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None
