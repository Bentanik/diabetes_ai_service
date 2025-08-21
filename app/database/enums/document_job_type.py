from enum import Enum


class DocumentJobType(str, Enum):
    """
    Enum định nghĩa các loại công việc xử lý tài liệu

    Values:
        UPLOAD: Xử lý tài liệu được upload
        TRAINING: Xử lý tài liệu training
    """
    UPLOAD = "upload_document"
    TRAINING = "training_document"
