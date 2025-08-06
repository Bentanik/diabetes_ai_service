from typing import List
import numpy as np
import asyncio
from core.llm import EmbeddingModel
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
        Đảm bảo model đã được khởi tạo
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
                self.logger.info(f"Embedding model đã sẵn sàng")
            except Exception as e:
                self.logger.error(
                    f"Lỗi khởi tạo Embedding model: {str(e)}", exc_info=True
                )
                raise

    async def embed_text(self, text: str) -> List[float]:
        """
        Tạo embedding cho một chuỗi text.

        Args:
            text (str): Chuỗi text cần tạo embedding.

        Returns:
            List[float]: Vector embedding của text.
        """
        await self._ensure_initialized()
        try:
            embedding = await self.model.embed(text)
            # Nếu model trả về batch, lấy phần tử đầu và chuyển sang list
            if isinstance(embedding, np.ndarray):
                if len(embedding.shape) == 2:
                    return embedding[0].tolist()
                else:
                    return embedding.tolist()
            elif isinstance(embedding, list):
                if isinstance(embedding[0], (list, np.ndarray)):
                    return list(embedding[0])
                else:
                    return list(embedding)
            else:
                return list(embedding)
        except Exception as e:
            self.logger.error(
                f"Lỗi khi tạo embedding cho text: {str(e)}", exc_info=True
            )
            raise

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Tạo embedding cho danh sách các chuỗi text.

        Args:
            texts (List[str]): Danh sách các chuỗi text cần tạo embedding.

        Returns:
            List[List[float]]: Ma trận embedding của các text.
        """
        await self._ensure_initialized()
        try:
            embeddings = await self.model.embed_batch(texts)
            # Chuyển về list of list
            if isinstance(embeddings, np.ndarray):
                return embeddings.tolist()
            elif isinstance(embeddings, list):
                # Đảm bảo mỗi phần tử là list[float]
                return [list(e) if not isinstance(e, list) else e for e in embeddings]
            else:
                # Trường hợp đặc biệt
                return list(embeddings)
        except Exception as e:
            self.logger.error(
                f"Lỗi khi tạo embedding cho list text: {str(e)}", exc_info=True
            )
            raise

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Tạo embedding cho danh sách các chuỗi text.

        Args:
            texts (List[str]): Danh sách các chuỗi text cần tạo embedding.

        Returns:
            List[List[float]]: Danh sách các vector embedding.
        """
        try:
            embeddings = await self.embed_texts(texts)
            return embeddings
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
        Tạo embedding cho một chuỗi query, trả về dạng list để tương thích với langchain.

        Args:
            text (str): Chuỗi query cần tạo embedding.

        Returns:
            List[float]: Vector embedding của query.
        """
        try:
            emb = await self.embed_text(text)
            return emb
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
