import asyncio
from typing import List, Optional, Tuple
import torch
from sentence_transformers import CrossEncoder
import logging

logger = logging.getLogger(__name__)

class RerankModel:
    _instance: Optional["RerankModel"] = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __new__(cls, model_name: str = "BAAI/bge-reranker-base", device: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model_name = model_name
            cls._instance.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            cls._instance.model = None
            cls._instance._is_loaded = False
        return cls._instance

    async def load(self) -> None:
        if self._is_loaded:
            return

        def _load():
            try:
                self.model = CrossEncoder(
                    self.model_name,
                    device=self.device,
                    trust_remote_code=True,
                    max_length=512
                )
            except Exception as e:
                logger.warning(f"Failed to load reranker on {self.device}: {e}")
                self.device = "cpu"
                self.model = CrossEncoder(
                    self.model_name,
                    device="cpu",
                    trust_remote_code=True,
                    max_length=512
                )

        await asyncio.to_thread(_load)
        self._is_loaded = True

    @classmethod
    async def get_instance(cls) -> "RerankModel":
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
                await cls._instance.load()
        return cls._instance

    async def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Re-rank danh sách documents theo độ liên quan với query.
        Trả về: [(document, score), ...] (giảm dần theo điểm)
        """
        if not self._is_loaded:
            await self.load()

        pairs = [[query, doc] for doc in documents]
        try:
            # Chạy async
            scores = await asyncio.to_thread(lambda: self.model.predict(pairs))
            if isinstance(scores, float):
                scores = [scores]
            # Ghép lại và sắp xếp
            results = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
            return results[:top_k]
        except torch.cuda.OutOfMemoryError:
            logger.warning("GPU OOM in rerank, switching to CPU...")
            # Tạm thời chuyển sang CPU
            original_device = self.model._target_device
            self.model._target_device = torch.device("cpu")
            scores = self.model.predict(pairs)
            self.model._target_device = original_device
            results = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
            return results[:top_k]
        except Exception as e:
            logger.error(f"Rerank failed: {e}")
            # Fallback: giữ nguyên thứ tự
            return [(doc, 0.0) for doc in documents[:top_k]]