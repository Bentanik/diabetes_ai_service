from dataclasses import dataclass
from core.cqrs.base import Query


@dataclass
class GetDocumentQuery(Query):
    """
    Lấy tài liệu theo ID

    Attributes:
        id: ID của tài liệu
    """

    id: str

    def __post_init__(self):
        if not self.id:
            raise ValueError("ID của tài liệu không được để trống")
