from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class BBox:
    """
    Bounding Box - Hộp giới hạn cho vùng text

    Attributes:
        left: Tọa độ x bên trái
        top: Tọa độ y phía trên
        right: Tọa độ x bên phải
        bottom: Tọa độ y phía dưới

    Lưu ý: Sử dụng hệ tọa độ top-left origin (gốc trên-trái)
    """

    left: float
    top: float
    right: float
    bottom: float

    @property
    def width(self) -> float:
        """Chiều rộng của bounding box"""
        return self.right - self.left

    @property
    def height(self) -> float:
        """Chiều cao của bounding box"""
        return self.bottom - self.top

    @property
    def area(self) -> float:
        """Diện tích của bounding box"""
        return self.width * self.height

    @property
    def center_x(self) -> float:
        """Tọa độ x của tâm"""
        return (self.left + self.right) / 2

    @property
    def center_y(self) -> float:
        """Tọa độ y của tâm"""
        return (self.top + self.bottom) / 2

    def overlaps_with(self, other: "BBox", threshold: float = 0.5) -> bool:
        """
        Kiểm tra xem bbox này có overlap với bbox khác không

        Args:
            other: BBox khác để so sánh
            threshold: Ngưỡng overlap (0.0 - 1.0)

        Returns:
            bool: True nếu overlap >= threshold
        """
        # Tính vùng giao nhau
        left = max(self.left, other.left)
        top = max(self.top, other.top)
        right = min(self.right, other.right)
        bottom = min(self.bottom, other.bottom)

        # Không có giao nhau
        if left >= right or top >= bottom:
            return False

        # Tính diện tích giao nhau
        intersection_area = (right - left) * (bottom - top)
        self_area = self.area

        if self_area == 0:
            return False

        # Tính tỷ lệ overlap
        overlap_ratio = intersection_area / self_area
        return overlap_ratio >= threshold

    def distance_to(self, other: "BBox") -> float:
        """
        Tính khoảng cách giữa 2 bbox (từ tâm đến tâm)

        Args:
            other: BBox khác

        Returns:
            float: Khoảng cách Euclidean
        """
        dx = self.center_x - other.center_x
        dy = self.center_y - other.center_y
        return (dx * dx + dy * dy) ** 0.5

    def is_horizontally_aligned(self, other: "BBox", tolerance: float = 5.0) -> bool:
        """
        Kiểm tra 2 bbox có cùng hàng ngang không (cùng độ cao y)

        Args:
            other: BBox khác
            tolerance: Dung sai cho phép (pixels)

        Returns:
            bool: True nếu cùng hàng ngang
        """
        return abs(self.center_y - other.center_y) <= tolerance

    def is_vertically_aligned(self, other: "BBox", tolerance: float = 5.0) -> bool:
        """
        Kiểm tra 2 bbox có cùng cột dọc không (cùng độ rộng x)

        Args:
            other: BBox khác
            tolerance: Dung sai cho phép (pixels)

        Returns:
            bool: True nếu cùng cột dọc
        """
        return abs(self.center_x - other.center_x) <= tolerance

    def expand(self, margin: float) -> "BBox":
        """
        Mở rộng bbox theo tất cả các hướng

        Args:
            margin: Kích thước mở rộng (pixels)

        Returns:
            BBox: BBox mới đã được mở rộng
        """
        return BBox(
            left=self.left - margin,
            top=self.top - margin,
            right=self.right + margin,
            bottom=self.bottom + margin,
        )

    def to_dict(self) -> dict:
        """
        Chuyển đổi sang dictionary để lưu JSON

        Returns:
            dict: Dictionary chứa tọa độ bbox
        """
        return {
            "left": self.left,
            "top": self.top,
            "right": self.right,
            "bottom": self.bottom,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BBox":
        """
        Tạo BBox từ dictionary

        Args:
            data: Dictionary chứa tọa độ

        Returns:
            BBox: BBox object mới
        """
        return cls(
            left=data["left"],
            top=data["top"],
            right=data["right"],
            bottom=data["bottom"],
        )

    @classmethod
    def merge_bboxes(cls, bboxes: List["BBox"]) -> Optional["BBox"]:
        """
        Gộp nhiều bbox thành một bbox bao quanh tất cả

        Args:
            bboxes: Danh sách các bbox cần gộp

        Returns:
            BBox: BBox bao quanh tất cả, hoặc None nếu list rỗng
        """
        if not bboxes:
            return None

        min_left = min(bbox.left for bbox in bboxes)
        min_top = min(bbox.top for bbox in bboxes)
        max_right = max(bbox.right for bbox in bboxes)
        max_bottom = max(bbox.bottom for bbox in bboxes)

        return cls(left=min_left, top=min_top, right=max_right, bottom=max_bottom)

    def __str__(self) -> str:
        """String representation của BBox"""
        return f"BBox(L={self.left:.1f}, T={self.top:.1f}, R={self.right:.1f}, B={self.bottom:.1f})"
