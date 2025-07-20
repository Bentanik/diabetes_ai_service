from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum
from datetime import datetime


class HealthStatus(str, Enum):
    """Trạng thái"""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"


class DatabaseHealth(BaseModel):
    """Model cho thông tin sức khỏe database"""

    status: HealthStatus = Field(..., description="Trạng thái database")
    connected: bool = Field(..., description="Kết nối thành công hay không")
    database: Optional[str] = Field(None, description="Tên database")
    collections: Optional[int] = Field(None, description="Số lượng collections", ge=0)
    data_size_mb: Optional[float] = Field(
        None, description="Kích thước dữ liệu (MB)", ge=0
    )
    storage_size_mb: Optional[float] = Field(
        None, description="Kích thước storage (MB)", ge=0
    )
    response_time_ms: Optional[float] = Field(
        None, description="Thời gian phản hồi (ms)", ge=0
    )
    error: Optional[str] = Field(None, description="Thông báo lỗi nếu có")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "healthy",
                    "connected": True,
                    "database": "diabetes_ai_dev",
                    "collections": 3,
                    "data_size_mb": 2.5,
                    "storage_size_mb": 5.2,
                    "response_time_ms": 15.3,
                    "error": None,
                },
                {
                    "status": "unhealthy",
                    "connected": False,
                    "database": None,
                    "collections": None,
                    "data_size_mb": None,
                    "storage_size_mb": None,
                    "response_time_ms": None,
                    "error": "Connection timeout",
                },
            ]
        }
    }


class ServiceInfo(BaseModel):
    """Model cho thông tin service"""

    name: str = Field(..., description="Tên service", min_length=1)
    version: str = Field(
        ..., description="Phiên bản service", pattern=r"^\d+\.\d+\.\d+$"
    )
    status: HealthStatus = Field(..., description="Trạng thái service")
    uptime_seconds: Optional[int] = Field(
        None, description="Thời gian hoạt động (giây)", ge=0
    )
    startup_time: Optional[datetime] = Field(None, description="Thời gian khởi động")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "Service Diabetes AI",
                    "version": "1.0.0",
                    "status": "healthy",
                    "uptime_seconds": 3600,
                    "startup_time": "2025-07-21T03:40:00Z",
                }
            ]
        }
    }


class HealthCheckData(BaseModel):
    """Model cho dữ liệu health check tổng thể"""

    status: HealthStatus = Field(..., description="Trạng thái tổng thể")
    service: ServiceInfo = Field(..., description="Thông tin service")
    database: DatabaseHealth = Field(..., description="Thông tin database")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "healthy",
                    "service": {
                        "name": "Service Diabetes AI",
                        "version": "1.0.0",
                        "status": "healthy",
                        "uptime_seconds": 3600,
                        "startup_time": "2025-07-21T03:40:00Z",
                    },
                    "database": {
                        "status": "healthy",
                        "connected": True,
                        "database": "diabetes_ai_dev",
                        "collections": 3,
                        "data_size_mb": 2.5,
                        "storage_size_mb": 5.2,
                        "response_time_ms": 15.3,
                        "error": None,
                    },
                }
            ]
        }
    }


class PingData(BaseModel):
    """Model cho ping response"""

    message: str = Field(..., description="Ping message", min_length=1)
    service: str = Field(..., description="Tên service", min_length=1)

    model_config = {
        "json_schema_extra": {
            "examples": [{"message": "pong", "service": "Service Diabetes AI"}]
        }
    }
