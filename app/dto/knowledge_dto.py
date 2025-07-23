from datetime import datetime
from pydantic import BaseModel

from app.database.models import KnowledgeModel


class KnowledgeDTO(BaseModel):
    id: str
    name: str
    description: str
    document_count: int
    total_size_bytes: int
    select_training: bool
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_model(cls, model: KnowledgeModel) -> "KnowledgeDTO":
        return cls(
            id=model.id,
            name=model.name,
            description=model.description,
            document_count=model.document_count,
            total_size_bytes=model.total_size_bytes,
            select_training=model.select_training,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )
