"""
Bounding Box - Value Object cho tọa độ khung chứa nội dung

File này định nghĩa BoundingBox để lưu trữ và xử lý thông tin về
vị trí và kích thước của một khung chứa nội dung trong trang tài liệu.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class BoundingBox:
    """
    Value Object chứa thông tin về tọa độ khung chứa nội dung

    Attributes:
        x0 (float): Tọa độ x điểm bắt đầu
        y0 (float): Tọa độ y điểm bắt đầu
        x1 (float): Tọa độ x điểm kết thúc
        y1 (float): Tọa độ y điểm kết thúc
    """
    x0: float = 0.0
    y0: float = 0.0
    x1: float = 0.0
    y1: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Chuyển đổi sang dictionary"""
        return {
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "BoundingBox":
        """Tạo instance từ dictionary"""
        return cls(
            x0=data.get("x0", 0),
            y0=data.get("y0", 0),
            x1=data.get("x1", 0),
            y1=data.get("y1", 0),
        ) 