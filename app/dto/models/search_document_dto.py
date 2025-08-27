from pydantic import BaseModel

from app.database.models import DocumentModel
from app.dto.models.document_model_dto import DocumentModelDTO

class SearchDocumentDTO(BaseModel):
    document: DocumentModelDTO
    content: str
    score: float

    @classmethod
    def from_model(cls, model: DocumentModel, content: str, score: float) -> "SearchDocumentDTO":
        """Tạo DTO từ model"""
        return cls(
            document=DocumentModelDTO.from_model(model),
            content=content,
            score=score,
        )
