from typing import Generic, TypeVar, Optional
from pydantic import BaseModel
from pydantic.fields import Field

T = TypeVar("T")


class ErrorResponse(BaseModel):
    """
    Schema cho response khi có lỗi xảy ra

    Attributes:
        detail (str): Mô tả chi tiết về lỗi
        errorCode (str): Mã lỗi để identify loại lỗi
        status (int): HTTP status code (400, 404, 500...)
        title (str): Tiêu đề ngắn gọn của lỗi

    Example:
        {
            "detail": "Validation failed for input data",
            "errorCode": "VALIDATION_ERROR",
            "status": 400,
            "title": "Bad Request"
        }
    """

    detail: str = Field(..., description="Mô tả chi tiết về lỗi")
    errorCode: str = Field(..., description="Mã lỗi để identify loại lỗi")
    status: int = Field(..., description="HTTP status code (400, 404, 500...)")
    title: str = Field(..., description="Tiêu đề ngắn gọn của lỗi")


class ErrorModel(BaseModel):
    """
    Schema cho thông tin lỗi

    Attributes:
        code (str): Mã lỗi ngắn gọn
        message (str): Thông điệp lỗi cho user

    Example:
        {
            "code": "INVALID_INPUT",
            "message": "Dữ liệu đầu vào không hợp lệ"
        }
    """

    code: str = Field(..., description="Mã lỗi ngắn gọn")
    message: str = Field(..., description="Thông điệp lỗi cho user")


class SuccessResponse(BaseModel, Generic[T]):
    """
    Schema cho response khi thành công với dữ liệu generic

    Attributes:
        isSuccess (bool): Trạng thái thành công
        code (str): Mã trạng thái thành công
        message (str): Thông điệp thành công
        data (Optional[T]): Dữ liệu trả về (có thể null)

    Example:
        {
            "isSuccess": true,
            "code": "SUCCESS",
            "message": "Xử lý thành công",
            "data": {...}  // Generic data type
        }
    """

    isSuccess: bool = Field(..., description="Trạng thái thành công")
    code: str = Field(..., description="Mã trạng thái thành công")
    message: str = Field(..., description="Thông điệp thành công")
    data: Optional[T] = Field(..., description="Dữ liệu trả về (có thể null)")
