"""
Document Type Enum - Enum cho loại tài liệu trong DTO

File này định nghĩa DocumentType enum cho DTO layer.
"""

from enum import Enum


class DocumentType(str, Enum):
    """
    Enum định nghĩa các loại tài liệu trong DTO

    Values:
        UPLOADED: Tài liệu được upload bởi người dùng
        TRAINING: Tài liệu dùng để training
        TRAINED: Tài liệu đã được training
    """

    UPLOADED = "uploaded_document"
    TRAINING = "training_document"
    TRAINED = "trained_document"
