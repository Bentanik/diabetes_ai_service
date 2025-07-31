from typing import Optional
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from utils import get_logger
import asyncio

logger = get_logger(__name__)

_reranker_instance: Optional[HuggingFaceCrossEncoder] = None
_reranker_model_name: str = "BAAI/bge-reranker-v2-m3"


async def get_reranker_model() -> HuggingFaceCrossEncoder:
    global _reranker_instance
    if _reranker_instance is None:
        logger.info("Đang tải reranker model...")
        try:
            _reranker_instance = await asyncio.to_thread(
                HuggingFaceCrossEncoder,
                model_name=_reranker_model_name
            )
        except Exception as e:
            logger.error(f"Lỗi tải reranker model: {str(e)}", exc_info=True)
            raise
    return _reranker_instance