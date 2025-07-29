"""
Knowledge Stats - Value Object cho thống kê của cơ sở tri thức

File này định nghĩa KnowledgeStats để lưu trữ và xử lý thông tin thống kê
về số lượng và dung lượng tài liệu trong cơ sở tri thức.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class KnowledgeStats:
    """
    Value Object chứa thông tin thống kê của cơ sở tri thức

    Attributes:
        document_count (int): Số lượng tài liệu
        total_size_bytes (int): Tổng dung lượng của các tài liệu (bytes)
    """

    document_count: int = 0
    total_size_bytes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi sang dictionary cho MongoDB"""
        return {
            "document_count": self.document_count,
            "total_size_bytes": self.total_size_bytes,
        }
