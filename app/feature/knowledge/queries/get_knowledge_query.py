from dataclasses import dataclass
from core.cqrs.base import Query


@dataclass
class GetKnowledgeQuery(Query):
    """
    Lấy cơ sở tri thức theo ID

    Attributes:
        id: ID của cơ sở tri thức
    """

    id: str

    def __post_init__(self):
        if not self.id:
            raise ValueError("ID của cơ sở tri thức không được để trống")
