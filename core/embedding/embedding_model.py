import asyncio
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import torch

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
            self.model = SentenceTransformer(
                self.model_name, 
                device=self.device, 
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
        return await asyncio.to_thread(lambda: self.model.encode(text, convert_to_numpy=True).tolist())

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not self._is_loaded:
            await self.load()
        return await asyncio.to_thread(lambda: self.model.encode(texts, convert_to_numpy=True).tolist())
    