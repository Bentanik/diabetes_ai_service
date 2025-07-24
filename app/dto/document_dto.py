from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from pydantic import Field
from app.database.models import DocumentModel, DocumentType


class DocumentDTO(BaseModel):
    id: str = Field(..., alias="_id")
    knowledge_id: str
    title: str
    description: Optional[str] = ""
    file_path: str
    file_size_bytes: int
    file_hash: Optional[str] = None
    type: DocumentType
    priority_diabetes: float
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @classmethod
    def from_model(cls, document: DocumentModel) -> "DocumentDTO":
        return cls(
            _id=document.id,
            knowledge_id=str(document.knowledge_id),
            title=document.title,
            description=document.description,
            file_path=document.file_path,
            file_size_bytes=document.file_size_bytes,
            file_hash=document.file_hash,
            type=document.type,
            priority_diabetes=document.priority_diabetes,
            created_at=document.created_at,
            updated_at=document.updated_at,
        )
