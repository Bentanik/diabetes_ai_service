"""
Bounding Box DTO - Value Object DTO cho tọa độ khung chứa nội dung

File này định nghĩa BoundingBoxDTO để chuyển đổi dữ liệu
giữa BoundingBox value object và API responses.
"""

from pydantic import BaseModel, Field

from app.database.value_objects import BoundingBox


class BoundingBoxDTO(BaseModel):
    """
    DTO cho tọa độ khung chứa nội dung

    Attributes:
        x0 (float): Tọa độ x điểm bắt đầu
        y0 (float): Tọa độ y điểm bắt đầu
        x1 (float): Tọa độ x điểm kết thúc
        y1 (float): Tọa độ y điểm kết thúc
    """
    x0: float = Field(0.0, description="Tọa độ x điểm bắt đầu")
    y0: float = Field(0.0, description="Tọa độ y điểm bắt đầu")
    x1: float = Field(0.0, description="Tọa độ x điểm kết thúc")
    y1: float = Field(0.0, description="Tọa độ y điểm kết thúc")

    @classmethod
    def from_value_object(cls, value_object: BoundingBox) -> "BoundingBoxDTO":
        """Tạo DTO từ value object"""
        return cls(
            x0=value_object.x0,
            y0=value_object.y0,
            x1=value_object.x1,
            y1=value_object.y1
        )

    def to_value_object(self) -> BoundingBox:
        """Chuyển đổi DTO thành value object"""
        return BoundingBox(
            x0=self.x0,
            y0=self.y0,
            x1=self.x1,
            y1=self.y1
        ) 