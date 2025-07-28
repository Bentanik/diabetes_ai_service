"""
Document Job Type Enum - Enum cho loại công việc xử lý tài liệu trong DTO

File này định nghĩa DocumentJobType enum cho DTO layer.
"""

from enum import Enum


class DocumentJobType(str, Enum):
    """
    Enum định nghĩa các loại công việc xử lý tài liệu trong DTO

    Values:
        UPLOAD: Xử lý tài liệu được upload
        TRAINING: Xử lý tài liệu training
    """
    UPLOAD = "upload_document"
    TRAINING = "training_document" 