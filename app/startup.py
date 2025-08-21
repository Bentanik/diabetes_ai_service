from contextlib import asynccontextmanager
from fastapi import FastAPI
from dotenv import load_dotenv
from app.database import initialize_database, close_mongodb_connection
from app.nlp import DiabetesClassifier
from app.storage import MinioManager
from app.worker import worker_start_all, worker_stop_all
from app.config import MinioConfig
from core.embedding import EmbeddingModel
from rag.vector_store.client import VectorStoreClient
from utils import get_logger

load_dotenv()

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Quản lý lifecycle của ứng dụng"""
    logger.info("Service Diabetes AI đang khởi động...")
    logger.info(f"{app.title} v{app.version}")

    try:
        # Tải model embedding
        await EmbeddingModel.get_instance()

        # Tải model classifier
        DiabetesClassifier()

        # Khởi tạo các worker
        worker_start_all()

        # Khởi tạo database và storage
        await initialize_database()
        MinioManager.get_instance().create_bucket_if_not_exists(
            MinioConfig.DOCUMENTS_BUCKET
        )

        # Khởi tạo Vector Store
        VectorStoreClient().connection

        yield

    except Exception as e:
        logger.error(f"Lỗi khởi động: {e}")
        raise
    finally:
        # Shutdown
        logger.info("Service đang tắt...")
        await worker_stop_all()
        await close_mongodb_connection()
        logger.info("Hoàn tất!")
