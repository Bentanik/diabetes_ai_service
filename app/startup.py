from contextlib import asynccontextmanager
from fastapi import FastAPI
from dotenv import load_dotenv
from app.database import initialize_database, close_mongodb_connection
from app.database.manager import get_collections
from app.database.models.setting_model import SettingModel
from app.storage import MinioManager
from app.worker import worker_start_all, worker_stop_all
from app.config import MinioConfig
from core.embedding import EmbeddingModel
from core.llm.gemini import GeminiClient
from rag.vector_store import VectorStoreOperations
from utils import get_logger, get_scorer_async
from rag.embedding import Embedding

load_dotenv()

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Quản lý lifecycle của ứng dụng"""
    logger.info("Service Diabetes AI đang khởi động...")
    logger.info(f"{app.title} v{app.version}")

    try:
        # Tải model
        await EmbeddingModel.get_instance()

        # Khởi tạo scorer
        await get_scorer_async()

        # Khởi tạo các worker
        worker_start_all()

        # Khởi tạo embedding
        await Embedding()._ensure_initialized()

        # Khởi tạo database và storage
        await initialize_database()
        MinioManager.get_instance().create_bucket_if_not_exists(
            MinioConfig.DOCUMENTS_BUCKET
        )

        # Khởi tạo Vector Store
        vector_store_operations = VectorStoreOperations.get_instance()
        available_collections = vector_store_operations.get_available_collections()
        logger.info(f"Đã khởi tạo Qdrant với các collection: {available_collections}")

        await init_setting()

        logger.info("Tất cả hệ thống đã sẵn sàng!")

        yield  # Ứng dụng đang chạy

        # Khởi tạo cài đặt

    except Exception as e:
        logger.error(f"Lỗi khởi động: {e}")
        raise
    finally:
        # Shutdown
        logger.info("Service đang tắt...")
        await worker_stop_all()
        await close_mongodb_connection()
        logger.info("Hoàn tất!")


async def init_setting():
    collections = get_collections()
    setting = await collections.settings.find_one({})
    if setting:
        return

    setting_model = SettingModel(
        top_k=5,
        search_accuracy=0.7,
        temperature=0.5,
        max_tokens=1000,
        system_prompt="Bạn là một AI hỗ trợ trong lĩnh vực y tế, đặc biệt là lĩnh vực đái tháo đường",
        context_prompt="Dựa vào những tài liệu được cung cấp hãy trả lời một cách đúng đắn và chính xác",
        list_knowledge_ids=[],
    )

    GeminiClient()
    await collections.settings.find_one_and_update(
        {}, {"$set": setting_model.to_dict()}, upsert=True
    )
