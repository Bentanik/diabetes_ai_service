"""
Document Job File - Value Object cho thông tin file của tài liệu

File này định nghĩa DocumentJobFile để lưu trữ và xử lý thông tin về file
của tài liệu trong hệ thống.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class DocumentFile:
    """
    Value Object chứa thông tin về file của document job

    Attributes:
        path (str): Đường dẫn đến file trong storage
        size_bytes (int): Kích thước file tính bằng bytes
        hash (Optional[str]): Hash của file để kiểm tra trùng lặp
    """

    path: str = ""
    size_bytes: int = 0
    name: str = ""
    type: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi sang dictionary cho MongoDB"""
        return {
            "file_path": self.path,
            "file_size_bytes": self.size_bytes,
            "file_name": self.name,
            "file_type": self.type,
        }
