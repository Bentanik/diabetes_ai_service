from typing import List
import numpy as np
import asyncio
from core.embedding import EmbeddingModel
import logging


class Embedding:
    def __init__(self):
        """
        Khởi tạo class Embedding để tạo embedding từ text hoặc list text.
        """
        self.logger = logging.getLogger(__name__)
        self.model = None
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def _ensure_initialized(self):
        """
        Đảm bảo model đã được khởi tạo.
        """
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            try:
                self.logger.info("Khởi tạo Embedding model...")
                self.model = await EmbeddingModel.get_instance()
                self._initialized = True
                self.logger.info("Embedding model đã sẵn sàng.")
            except Exception as e:
                self.logger.error(
                    f"Lỗi khởi tạo Embedding model: {str(e)}", exc_info=True
                )
                raise

    async def embed_text(self, text: str) -> List[float]:
        """
        Tạo embedding cho một chuỗi text.
        """
        await self._ensure_initialized()
        try:
            embedding = await self.model.embed(text)
            return self._normalize_embedding(embedding)
        except Exception as e:
            self.logger.error(
                f"Lỗi khi tạo embedding cho text: {str(e)}", exc_info=True
            )
            raise

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Tạo embedding cho danh sách các chuỗi text.
        """
        await self._ensure_initialized()
        try:
            embeddings = await self.model.embed_batch(texts)
            if isinstance(embeddings, np.ndarray):
                return embeddings.tolist()
            elif isinstance(embeddings, list):
                return [list(e) if not isinstance(e, list) else e for e in embeddings]
            else:
                return list(embeddings)
        except Exception as e:
            self.logger.error(
                f"Lỗi khi tạo embedding cho list text: {str(e)}", exc_info=True
            )
            raise

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Tạo embedding cho danh sách documents.
        """
        try:
            return await self.embed_texts(texts)
        except RuntimeError as e:
            if "cannot schedule new coroutine" in str(e):
                raise RuntimeError(
                    "Cannot call embed_documents in an async context. Use async_embed_texts instead."
                )
            raise
        except Exception as e:
            self.logger.error(
                f"Lỗi khi tạo embeddings cho documents: {str(e)}", exc_info=True
            )
            raise

    async def embed_query(self, text: str) -> List[float]:
        """
        Tạo embedding cho một query.
        """
        try:
            return await self.embed_text(text)
        except RuntimeError as e:
            if "cannot schedule new coroutine" in str(e):
                raise RuntimeError(
                    "Cannot call embed_query in an async context. Use async_embed_text instead."
                )
            raise
        except Exception as e:
            self.logger.error(
                f"Lỗi khi tạo embedding cho query: {str(e)}", exc_info=True
            )
            raise

    def _normalize_embedding(self, embedding) -> List[float]:
        """
        Normalize đầu ra embedding thành list[float]
        """
        if isinstance(embedding, np.ndarray):
            return embedding[0].tolist() if embedding.ndim == 2 else embedding.tolist()
        elif isinstance(embedding, list):
            return (
                list(embedding[0])
                if isinstance(embedding[0], (list, np.ndarray))
                else list(embedding)
            )
        else:
            return list(embedding)
