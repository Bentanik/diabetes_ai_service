from typing import Generic, TypeVar, Optional
from pydantic import BaseModel, Field

T = TypeVar("T")


class SuccessValue(BaseModel, Generic[T]):
    """Schema cho value khi thành công"""

    code: str = Field(..., description="Mã trạng thái thành công")
    message: str = Field(..., description="Thông điệp thành công")
    data: Optional[T] = Field(None, description="Dữ liệu trả về (có thể null)")


class SuccessResponse(BaseModel, Generic[T]):
    """Schema cho response khi thành công"""

    isSuccess: bool = Field(..., description="Trạng thái thành công")
    value: SuccessValue[T]


class ErrorResponse(BaseModel):
    """Schema cho response khi có lỗi xảy ra"""

    detail: str = Field(..., description="Mô tả chi tiết về lỗi")
    errorCode: str = Field(..., description="Mã lỗi để identify loại lỗi")
    status: int = Field(..., description="HTTP status code (400, 404, 500...)")
    title: str = Field(..., description="Tiêu đề ngắn gọn của lỗi")


class ErrorModel(BaseModel):
    """Schema cho thông tin lỗi"""

    code: str = Field(..., description="Mã lỗi ngắn gọn")
    message: str = Field(..., description="Thông điệp lỗi cho user")
