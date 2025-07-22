from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.database import initialize_database, close_mongodb_connection
from utils import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Quản lý lifecycle của ứng dụng"""
    logger.info("Service Diabetes AI đang khởi động...")
    logger.info(f"{app.title} v{app.version}")

    try:
        # Khởi tạo database
        await initialize_database()
        logger.info("Tất cả hệ thống đã sẵn sàng!")

        yield  # Ứng dụng đang chạy

    except Exception as e:
        logger.error(f"Lỗi khởi động: {e}")
        raise
    finally:
        # Shutdown
        logger.info("Service đang tắt...")
        await close_mongodb_connection()
        logger.info("Hoàn tất!")
