import asyncio
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import torch
import logging

logger = logging.getLogger(__name__)

class EmbeddingModel:
    _instance: Optional["EmbeddingModel"] = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __new__(cls, model_name: str = "Alibaba-NLP/gte-multilingual-base", device: Optional[str] = None):
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
                self.model = SentenceTransformer(
                    self.model_name, 
                    device=self.device, 
                    trust_remote_code=True
                )
            except Exception as e:
                logger.warning(f"Failed to load on {self.device}: {e}")
                # Fallback to CPU
                self.device = "cpu"
                self.model = SentenceTransformer(
                    self.model_name, 
                    device="cpu", 
                    trust_remote_code=True
                )

        await asyncio.to_thread(_load)
        self._is_loaded = True

    @classmethod
    async def get_instance(cls) -> "EmbeddingModel":
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
                await cls._instance.load()
        return cls._instance

    async def embed(self, text: str) -> List[float]:
        if not self._is_loaded:
            await self.load()
        
        try:
            result = await asyncio.to_thread(lambda: self.model.encode(text, convert_to_numpy=True).tolist())
            # Cleanup GPU memory
            if self.device == "cuda":
                torch.cuda.empty_cache()
            return result
        except torch.cuda.OutOfMemoryError:
            logger.warning("GPU OOM, retrying...")
            torch.cuda.empty_cache()
            return await asyncio.to_thread(lambda: self.model.encode(text, convert_to_numpy=True).tolist())

    async def embed_batch(self, texts: List[str], max_batch_size: int = 5) -> List[List[float]]:
        if not self._is_loaded:
            await self.load()
        
        # Split into smaller batches for GPU safety
        if len(texts) > max_batch_size:
            all_embeddings = []
            for i in range(0, len(texts), max_batch_size):
                batch = texts[i:i + max_batch_size]
                batch_embeddings = await self.embed_batch(batch, max_batch_size)
                all_embeddings.extend(batch_embeddings)
            return all_embeddings
        
        try:
            result = await asyncio.to_thread(lambda: self.model.encode(texts, convert_to_numpy=True).tolist())
            if self.device == "cuda":
                torch.cuda.empty_cache()
            return result
        except torch.cuda.OutOfMemoryError:
            logger.warning("GPU OOM in batch, processing individually...")
            # Fallback to individual processing
            return [await self.embed(text) for text in texts]
