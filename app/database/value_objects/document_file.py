"""
Document File - Value Object cho thông tin file của tài liệu

File này định nghĩa DocumentFile để lưu trữ và xử lý thông tin về file
của tài liệu trong hệ thống.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class DocumentFile:
    """
    Value Object chứa thông tin về file của document

    Attributes:
        path (str): Đường dẫn đến file trong storage
        size_bytes (int): Kích thước file tính bằng bytes
        hash (Optional[str]): Hash của file để kiểm tra trùng lặp
    """

    path: str = ""
    size_bytes: int = 0
    hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi sang dictionary cho MongoDB"""
        return {
            "file_path": self.path,
            "file_size_bytes": self.size_bytes,
            "file_hash": self.hash,
        }
