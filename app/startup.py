from contextlib import asynccontextmanager
from fastapi import FastAPI
from dotenv import load_dotenv
from app.database import initialize_database, close_mongodb_connection
from app.storage import MinioManager
from app.worker import worker_start_all, worker_stop_all
from app.config import MinioConfig
from core.llm import get_embedding_model
from rag.chunking import get_chunking_instance
from rag.vector_store import VectorStoreManager
from utils import get_logger, get_scorer_async

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

        # Khởi tạo scorer
        await get_scorer_async()

        # Khởi tạo Chunking
        await get_chunking_instance(enable_caching=True)

        # Khởi tạo các worker
        worker_start_all()

        # Khởi tạo database và storage
        await initialize_database()
        MinioManager.get_instance().create_bucket_if_not_exists(
            MinioConfig.DOCUMENTS_BUCKET
        )

        # Khởi tạo Qdrant VectorStoreManager
        vector_store_manager = VectorStoreManager()
        available_collections = vector_store_manager.get_collection_names()
        logger.info(f"Đã khởi tạo Qdrant với các collection: {available_collections}")

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
