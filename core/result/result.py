from typing import Generic, TypeVar, Optional, Any
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse
import json
from datetime import datetime, date

from .models import ErrorModel
from .enums import ResultStatus

T = TypeVar("T")


class Result(BaseModel, Generic[T]):
    """Pattern Result cho xử lý response thống nhất"""

    is_success: bool = Field(..., description="Trạng thái thành công")
    data: Optional[T] = Field(None, description="Dữ liệu khi thành công")
    error: Optional[ErrorModel] = Field(None, description="Lỗi khi thất bại")
    status: ResultStatus = Field(..., description="Trạng thái result")
    code: Optional[str] = Field(None, description="Mã code cho success")
    message: Optional[str] = Field(None, description="Thông báo cho success")

    # Mapping status sang HTTP response
    _STATUS_MAP = {
        ResultStatus.SUCCESS: (200, "OK"),
        ResultStatus.ERROR: (400, "Bad Request"),
        ResultStatus.VALIDATION_ERROR: (400, "Bad Request"),
        ResultStatus.NOT_FOUND: (404, "Not Found"),
        ResultStatus.UNAUTHORIZED: (401, "Unauthorized"),
        ResultStatus.FORBIDDEN: (403, "Forbidden"),
        ResultStatus.INTERNAL_ERROR: (500, "Internal Server Error"),
        ResultStatus.SERVICE_UNAVAILABLE: (503, "Service Unavailable"),
    }

    @staticmethod
    def _serialize_data(data: Any) -> Any:
        """Chuyển đổi data thành format có thể serialize JSON"""
        if data is None:
            return None

        # Xử lý datetime objects trước tiên
        if isinstance(data, (datetime, date)):
            return data.isoformat()

        # Nếu là Pydantic model
        if hasattr(data, "model_dump"):
            serialized = data.model_dump()
            # Recursive serialize để xử lý datetime bên trong
            return Result._serialize_data(serialized)

        # Nếu là list
        if isinstance(data, list):
            return [Result._serialize_data(item) for item in data]

        # Nếu là dict, serialize recursive
        if isinstance(data, dict):
            return {key: Result._serialize_data(value) for key, value in data.items()}

        # Kiểm tra có thể serialize JSON không
        try:
            json.dumps(data)
            return data
        except (TypeError, ValueError):
            # Nếu không serialize được thì convert sang string
            return str(data)

    @classmethod
    def success(
        cls,
        data: T = None,
        code: str = "SUCCESS",
        message: str = "Thành công",
        status: ResultStatus = ResultStatus.SUCCESS,
    ) -> "Result[T]":
        """Tạo Result thành công"""
        return cls(
            is_success=True,
            data=data,
            error=None,
            status=status,
            code=code,
            message=message,
        )

    @classmethod
    def failure(
        cls,
        code: str,
        message: str,
        status: ResultStatus = ResultStatus.ERROR,
    ) -> "Result[T]":
        """Tạo Result thất bại"""
        error = ErrorModel(code=code, message=message)
        return cls(
            is_success=False,
            data=None,
            error=error,
            status=status,
            code=None,
            message=None,
        )

    @classmethod
    def service_unavailable(
        cls, code: str, message: str, data: T = None
    ) -> "Result[T]":
        """Tạo service unavailable error"""
        error = ErrorModel(code=code, message=message)
        return cls(
            is_success=False,
            data=data,
            error=error,
            status=ResultStatus.SERVICE_UNAVAILABLE,
            code=None,
            message=None,
        )

    @classmethod
    def validation_error(cls, code: str, message: str) -> "Result[T]":
        """Tạo validation error"""
        return cls.failure(code, message, ResultStatus.VALIDATION_ERROR)

    @classmethod
    def not_found(cls, code: str, message: str) -> "Result[T]":
        """Tạo not found error"""
        return cls.failure(code, message, ResultStatus.NOT_FOUND)

    @classmethod
    def unauthorized(cls, code: str, message: str) -> "Result[T]":
        """Tạo unauthorized error"""
        return cls.failure(code, message, ResultStatus.UNAUTHORIZED)

    @classmethod
    def forbidden(cls, code: str, message: str) -> "Result[T]":
        """Tạo forbidden error"""
        return cls.failure(code, message, ResultStatus.FORBIDDEN)

    @classmethod
    def internal_error(cls, code: str, message: str) -> "Result[T]":
        """Tạo internal server error"""
        return cls.failure(code, message, ResultStatus.INTERNAL_ERROR)

    @classmethod
    def bad_request(cls, code: str, message: str) -> "Result[T]":
        """Tạo bad request error"""
        return cls.failure(code, message, ResultStatus.ERROR)

    def to_response(self) -> JSONResponse:
        """Chuyển đổi Result thành JSONResponse"""
        if self.is_success:
            return self._success_response()
        else:
            return self._error_response()

    def _success_response(self) -> JSONResponse:
        """Tạo response theo SuccessResponse format"""
        http_status, _ = self._STATUS_MAP[self.status]

        serialized_data = self._serialize_data(self.data)

        content = {
            "isSuccess": True,
            "code": self.code,
            "message": self.message,
            "data": serialized_data,
        }

        return JSONResponse(status_code=http_status, content=content)

    def _error_response(self) -> JSONResponse:
        """Tạo response theo ErrorResponse format"""
        http_status, title = self._STATUS_MAP.get(self.status, (400, "Error"))

        content = {
            "detail": self.error.message,
            "errorCode": self.error.code,
            "status": http_status,
            "title": title,
        }

        # Nếu có thêm data (optional)
        if self.data is not None:
            content["data"] = self._serialize_data(self.data)

        return JSONResponse(status_code=http_status, content=content)
