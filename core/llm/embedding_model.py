import time
from typing import Optional
from sentence_transformers import SentenceTransformer
from utils import get_logger

logger = get_logger(__name__)

_model_instance: Optional[SentenceTransformer] = None
_model_name: str = "intfloat/multilingual-e5-base"


def get_embedding_model() -> SentenceTransformer:
    global _model_instance
    if _model_instance is None:
        logger.info("Đang tải embedding model...")
        start_time = time.time()
        _model_instance = SentenceTransformer(_model_name)
        logger.info(
            f"Tải embedding model thành công trong {time.time() - start_time:.2f} giây"
        )
    return _model_instance
