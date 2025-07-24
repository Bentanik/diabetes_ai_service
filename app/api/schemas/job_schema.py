from enum import Enum
from pydantic import BaseModel


class DocumentJobType(str, Enum):
    UPLOAD = "upload_document"
    TRAINING = "training_document"


class DocumentJobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
