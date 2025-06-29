"""Điểm khởi đầu của ứng dụng FastAPI với tài liệu Swagger được cải thiện."""

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from api.router import router as api_router
from config.settings import config
from core.logging_config import get_logger

logger = get_logger(__name__)

# Mô tả ứng dụng được cải thiện
description = """
🏥 **Dịch Vụ AI Quản Lý Chăm Sóc Tiểu Đường**

API này cung cấp các dịch vụ thông minh cho việc tạo kế hoạch chăm sóc tiểu đường và phân tích kết quả đo.

## Tính Năng

* **🎯 Tạo Kế Hoạch Chăm Sóc**: Tạo lịch đo thông số y tế cá nhân hóa dựa trên dữ liệu bệnh nhân
* **📊 Phân Tích Kết Quả Đo**: Phân tích kết quả đo của bệnh nhân với phản hồi được hỗ trợ bởi AI
* **🔍 Theo Dõi Sức Khỏe**: Theo dõi và giám sát các chỉ số sức khỏe của bệnh nhân

## Giới Hạn Tốc Độ

Hiện tại chưa có giới hạn tốc độ nào được áp dụng.
"""

# Tạo ứng dụng FastAPI với cấu hình được cải thiện
app = FastAPI(
    title=config["app_title"],
    version=config["app_version"],
    description=description,
    docs_url="/docs",
    redoc_url="/redoc",
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    servers=[
        {"url": "http://localhost:8000", "description": ""},
    ],
)

# Bao gồm các route API
app.include_router(api_router, prefix="/api/v1")


@app.on_event("startup")
async def startup_event():
    """Xử lý sự kiện khởi động ứng dụng."""
    logger.info(f"Đang khởi động {config['app_title']} v{config['app_version']}")
    logger.info("Khởi động ứng dụng hoàn tất")


@app.on_event("shutdown")
async def shutdown_event():
    """Xử lý sự kiện tắt ứng dụng."""
    logger.info("Bắt đầu tắt ứng dụng")


@app.get(
    "/health",
    summary="🏥 Kiểm Tra Sức Khỏe",
    description="Kiểm tra trạng thái hoạt động của dịch vụ AI",
    tags=["Hệ Thống"],
    responses={
        200: {
            "description": "Dịch vụ hoạt động bình thường",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "service": "AI Service - CarePlan Generator",
                        "version": "1.0.0",
                    }
                }
            },
        }
    },
)
async def health_check():
    """Endpoint kiểm tra sức khỏe để giám sát trạng thái dịch vụ."""
    return {
        "status": "healthy",
        "service": config["app_title"],
        "version": config["app_version"],
    }
