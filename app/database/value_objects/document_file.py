"""
Document File - Value Object cho thông tin file của tài liệu

File này định nghĩa DocumentFile để lưu trữ và xử lý thông tin về file
của tài liệu trong hệ thống.
"""

from dataclasses import dataclass
from typing import Dict, Union, Optional


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

    def to_dict(self) -> Dict[str, Union[str, int, None]]:
        """
        Chuyển đổi thành dictionary

        Returns:
            Dict: Thông tin file dưới dạng dictionary
        """
        return {
            "file_path": self.path,
            "file_size_bytes": self.size_bytes,
            "file_hash": self.hash
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Union[str, int, None]]) -> "DocumentFile":
        """
        Tạo DocumentFile từ dictionary

        Args:
            data: Dictionary chứa thông tin file

        Returns:
            DocumentFile: Instance mới của DocumentFile
        """
        return cls(
            path=data.get("file_path", ""),
            size_bytes=data.get("file_size_bytes", 0),
            hash=data.get("file_hash")
        )
