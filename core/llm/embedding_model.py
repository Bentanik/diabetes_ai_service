from typing import Optional
from langchain_huggingface import HuggingFaceEmbeddings
from utils import get_logger

logger = get_logger(__name__)

_model_instance: Optional[HuggingFaceEmbeddings] = None
_model_name: str = "intfloat/multilingual-e5-base"


def get_embedding_model() -> HuggingFaceEmbeddings:
    global _model_instance
    if _model_instance is None:
        logger.info("Đang tải embedding model...")
        _model_instance = HuggingFaceEmbeddings(model_name=_model_name)
    return _model_instance
