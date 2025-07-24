from typing import Dict, Union
from enum import Enum
from datetime import datetime
from app.database.models import BaseModel

DocumentJobDict = Dict[str, Union[str, int, bool, float, datetime, None]]


class DocumentJobType(str, Enum):
    UPLOAD = "upload_document"
    TRAINING = "training_document"


class DocumentJobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentJobModel(BaseModel):
    def __init__(
        self,
        document_id: str,
        knowledge_id: str,
        title: str,
        description: str,
        file_path: str,
        type: DocumentJobType = DocumentJobType.UPLOAD,
        is_diabetes: bool = False,
        status: DocumentJobStatus = DocumentJobStatus.PENDING,
        progress: float = 0.0,
        progress_message: str = "",
        priority_diabetes: float = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.document_id = document_id
        self.knowledge_id = knowledge_id
        self.title = title
        self.description = description
        self.file_path = file_path
        self.type = type
        self.is_diabetes = is_diabetes
        self.status = status
        self.progress = progress
        self.progress_message = progress_message
        self.priority_diabetes = priority_diabetes

    def to_dict(self) -> DocumentJobDict:
        result = super().to_dict()
        result.update(
            {
                "document_id": self.document_id,
                "knowledge_id": self.knowledge_id,
                "title": self.title,
                "description": self.description,
                "file_path": self.file_path,
                "type": self.type.value if isinstance(self.type, Enum) else self.type,
                "is_diabetes": self.is_diabetes,
                "status": (
                    self.status.value if isinstance(self.status, Enum) else self.status
                ),
                "progress": self.progress,
                "progress_message": self.progress_message,
                "priority_diabetes": self.priority_diabetes,
            }
        )
        return result

    @classmethod
    def from_dict(cls, data: DocumentJobDict) -> "DocumentJobModel":
        data = dict(data)
        type_val = data.pop("type", DocumentJobType.UPLOAD)
        if isinstance(type_val, str):
            type_val = DocumentJobType(type_val)

        status_val = data.pop("status", DocumentJobStatus.PENDING)
        if isinstance(status_val, str):
            status_val = DocumentJobStatus(status_val)

        return cls(
            document_id=data.pop("document_id"),
            knowledge_id=data.pop("knowledge_id"),
            title=data.pop("title"),
            description=data.pop("description"),
            file_path=data.pop("file_path"),
            type=type_val,
            is_diabetes=data.pop("is_diabetes", False),
            status=status_val,
            progress=data.pop("progress", 0.0),
            progress_message=data.pop("progress_message", ""),
            priority_diabetes=data.pop("priority_diabetes", 0.0),
            **data,
        )
