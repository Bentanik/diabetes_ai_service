from typing import List, Optional
import asyncio
from core.llm import get_reranker_model
from utils import get_logger
from pydantic import BaseModel

logger = get_logger(__name__)

class RerankResult(BaseModel):
    text: str
    score: float

    class Config:
        json_encoders = {
            float: lambda v: float(v)
        }

class Reranker:
    _instance: Optional["Reranker"] = None
    _initialized: bool = False
    _init_lock = asyncio.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
        return cls._instance

    async def _ensure_initialized(self):
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            try:
                logger.info("Khởi tạo Reranker model...")
                self._model = await get_reranker_model()
                self._initialized = True
                logger.info("Reranker model đã sẵn sàng")
            except Exception as e:
                logger.error(f"Lỗi khởi tạo Reranker model: {str(e)}", exc_info=True)
                raise

    async def rerank(self, query: str, documents: List[str], top_k: int = 10) -> List[RerankResult]:
        await self._ensure_initialized()
        if not documents:
            return []
        try:
            pairs = [(query, doc) for doc in documents]
            scores = await asyncio.to_thread(self._model.score, pairs)
            reranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
            return [RerankResult(text=doc, score=score) for doc, score in reranked[:top_k]]
        except Exception as e:
            logger.error(f"Lỗi khi rerank documents: {str(e)}", exc_info=True)
            raise