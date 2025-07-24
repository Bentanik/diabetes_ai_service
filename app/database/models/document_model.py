from enum import Enum
from typing import Dict, Union, Optional
from bson import ObjectId
from datetime import datetime

from app.database.models import BaseModel

DocumentDict = Dict[str, Union[str, int, bool, datetime, ObjectId, None]]


class DocumentType(str, Enum):
    UPLOAD = "upload_document"
    TRAINING = "training_document"


class DocumentModel(BaseModel):
    """
    Model cho Document (Tài liệu thuộc Knowledge)
    """

    def __init__(
        self,
        knowledge_id: ObjectId,
        title: str,
        description: Optional[str] = "",
        file_path: str = "",
        file_size_bytes: int = 0,
        file_hash: Optional[str] = None,
        type: DocumentType = DocumentType.UPLOAD,
        priority_diabetes: float = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.knowledge_id = knowledge_id
        self.title = title
        self.description = description
        self.file_path = file_path
        self.file_size_bytes = file_size_bytes
        self.file_hash = file_hash
        self.type = type
        self.priority_diabetes = priority_diabetes

    def to_dict(self) -> DocumentDict:
        result = super().to_dict()
        result.update(
            {
                "knowledge_id": self.knowledge_id,
                "title": self.title,
                "description": self.description,
                "file_path": self.file_path,
                "file_size_bytes": self.file_size_bytes,
                "file_hash": self.file_hash,
                "type": self.type,
                "priority_diabetes": self.priority_diabetes,
            }
        )
        return result

    @classmethod
    def from_dict(cls, data: DocumentDict) -> "DocumentModel":
        data = dict(data)
        knowledge_id = data.pop("knowledge_id")
        title = data.pop("title", "")
        description = data.pop("description", "")
        file_path = data.pop("file_path", "")
        file_size_bytes = data.pop("file_size_bytes", 0)
        file_hash = data.pop("file_hash", None)
        type = data.pop("type", DocumentType.UPLOAD)
        priority_diabetes = data.pop("priority_diabetes", 0.0)

        return cls(
            knowledge_id=knowledge_id,
            title=title,
            description=description,
            file_path=file_path,
            file_size_bytes=file_size_bytes,
            file_hash=file_hash,
            type=type,
            priority_diabetes=priority_diabetes,
            **data,
        )
