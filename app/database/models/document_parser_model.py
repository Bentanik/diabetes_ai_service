from enum import Enum
from typing import Dict, Union, Optional
from app.database.models import BaseModel

DocumentParserDict = Dict[str, Union[str, bool, dict, None]]


class DocumentType(str, Enum):
    UPLOAD = "upload_document"
    TRAINING = "training_document"


class BBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "BBox":
        return cls(
            x0=data.get("x0", 0),
            y0=data.get("y0", 0),
            x1=data.get("x1", 0),
            y1=data.get("y1", 0),
        )


class Metadata(BaseModel):
    source: str
    page: int
    bbox: BBox
    block_index: Optional[int] = None
    document_type: Optional[DocumentType] = DocumentType.UPLOAD

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "page": self.page,
            "bbox": self.bbox.to_dict(),
            "block_index": self.block_index,
            "document_type": self.document_type.value if self.document_type else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Metadata":
        return cls(
            source=data.get("source", ""),
            page=data.get("page", 0),
            bbox=BBox.from_dict(data.get("bbox", {})),
            block_index=data.get("block_index"),
            document_type=(
                DocumentType(data["document_type"])
                if data.get("document_type")
                else DocumentType.UPLOAD
            ),
        )


class DocumentParserModel(BaseModel):
    document_id: str
    content: str
    metadata: Metadata
    is_active: bool = True

    def to_dict(self) -> DocumentParserDict:
        result = super().to_dict()
        result.update(
            {
                "document_id": self.document_id,
                "content": self.content,
                "metadata": self.metadata.to_dict(),
                "is_active": self.is_active,
            }
        )
        return result

    @classmethod
    def from_dict(cls, data: DocumentParserDict) -> "DocumentParserModel":
        data = dict(data)
        document_id = data.pop("document_id", None)
        content = data.pop("content", "")
        metadata_dict = data.pop("metadata", {})
        metadata = Metadata.from_dict(metadata_dict)
        is_active = data.pop("is_active", True)
        return cls(
            document_id=document_id,
            content=content,
            metadata=metadata,
            is_active=is_active,
            **data,
        )
