import asyncio
from typing import Optional
from langchain_huggingface import HuggingFaceEmbeddings
from utils import get_logger

logger = get_logger(__name__)

_model_instance: Optional[HuggingFaceEmbeddings] = None
_model_name: str = "intfloat/multilingual-e5-base"

async def get_embedding_model() -> HuggingFaceEmbeddings:
    global _model_instance
    if _model_instance is None:
        logger.info("Đang tải embedding model...")
        try:
            _model_instance = await asyncio.to_thread(
                HuggingFaceEmbeddings,
                model_name=_model_name
            )
            logger.info(
                f"Tải embedding model thành công"
            )
        except Exception as e:
            logger.error(f"Lỗi tải embedding model: {str(e)}", exc_info=True)
            raise
    return _model_instance