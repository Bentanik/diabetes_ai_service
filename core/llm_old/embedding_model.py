# EmbeddingModel - Singleton sử dụng 1 model embedding cố định

import asyncio
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from typing import List, Optional


class EmbeddingModel:
    _instance: Optional["EmbeddingModel"] = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __init__(self) -> None:
        self.model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        self.model: Optional[HuggingFaceEmbeddings] = None
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None
        self._is_loaded = False

    async def load(self) -> None:
        if self._is_loaded:
            return

        def _load():
            self.model = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"device": "cuda"},
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

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
        return await asyncio.to_thread(self.model.embed_query, text)

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return await asyncio.to_thread(self.model.embed_documents, texts)

    def count_tokens(self, text: str) -> int:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer chưa được load. Gọi await load() trước.")
        return len(self.tokenizer.encode(text))
