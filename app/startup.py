from contextlib import asynccontextmanager
from fastapi import FastAPI
from dotenv import load_dotenv
from app.database import initialize_database, close_mongodb_connection
from app.storage import minio_manager
from app.worker import worker_start_all, worker_stop_all
from core.llm import get_embedding_model
from utils import get_logger, get_scorer

# Load environment variables from .env file
load_dotenv()

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Quản lý lifecycle của ứng dụng"""
    logger.info("Service Diabetes AI đang khởi động...")
    logger.info(f"{app.title} v{app.version}")

    try:
        # Tải model
        get_embedding_model()

        scorer = get_scorer()
        await scorer.precompute_embeddings()

        # Khởi tạo các worker
        worker_start_all()

        # Khởi tạo database
        await initialize_database()
        minio_manager.create_bucket_if_not_exists("documents")

        logger.info("Tất cả hệ thống đã sẵn sàng!")

        yield  # Ứng dụng đang chạy

    except Exception as e:
        logger.error(f"Lỗi khởi động: {e}")
        raise
    finally:
        # Shutdown
        logger.info("Service đang tắt...")
        await worker_stop_all()
        await close_mongodb_connection()
        logger.info("Hoàn tất!")
