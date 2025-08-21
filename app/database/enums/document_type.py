"""
Document Type - Enum định nghĩa các loại tài liệu trong hệ thống

File này định nghĩa các loại tài liệu có thể được sử dụng trong hệ thống,
phân biệt giữa tài liệu được upload bởi người dùng và tài liệu dùng để training.
"""

from enum import Enum


class DocumentType(str, Enum):
    """
    Enum chứa các loại tài liệu trong hệ thống.

    Values:
        UPLOAD (str): Tài liệu được người dùng upload lên hệ thống
        TRAINING (str): Tài liệu đang được xử lý training model
        TRAINED (str): Tài liệu đã được training
    """

    UPLOADED = "uploaded_document"
    TRAINING = "training_document"
    TRAINED = "trained_document"