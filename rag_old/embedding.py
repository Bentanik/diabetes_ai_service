from typing import List
import numpy as np
import asyncio
from core.llm import get_embedding_model
from utils import get_logger

class Embedding:
    def __init__(self):
        """
        Khởi tạo class Embedding để tạo embedding từ text hoặc list text.
        """
        self.logger = get_logger(__name__)
        self.model = None
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def _ensure_initialized(self):
        """
        Đảm bảo model đã được khởi tạo - thread-safe và async.
        """
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            try:
                self.logger.info("Khởi tạo Embedding model...")
                self.model = await get_embedding_model()
                self._initialized = True
                self.logger.info(f"Embedding model đã sẵn sàng")
            except Exception as e:
                self.logger.error(f"Lỗi khởi tạo Embedding model: {str(e)}", exc_info=True)
                raise

    async def embed_text(self, text: str) -> np.ndarray:
        """
        Tạo embedding cho một chuỗi text.

        Args:
            text (str): Chuỗi text cần tạo embedding.

        Returns:
            np.ndarray: Vector embedding của text.
        """
        await self._ensure_initialized()
        try:
            embedding = await asyncio.to_thread(
                self.model.encode, [text], show_progress_bar=False
            )
            return embedding[0]
        except Exception as e:
            self.logger.error(f"Lỗi khi tạo embedding cho text: {str(e)}", exc_info=True)
            raise

    async def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Tạo embedding cho danh sách các chuỗi text.

        Args:
            texts (List[str]): Danh sách các chuỗi text cần tạo embedding.

        Returns:
            np.ndarray: Ma trận embedding của các text (shape: [len(texts), vector_size]).
        """
        await self._ensure_initialized()
        try:
            embeddings = await asyncio.to_thread(
                self.model.encode, texts, show_progress_bar=False
            )
            return embeddings
        except Exception as e:
            self.logger.error(f"Lỗi khi tạo embedding cho list text: {str(e)}", exc_info=True)
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Tạo embedding cho danh sách các chuỗi text, trả về dạng list để tương thích với langchain.

        Args:
            texts (List[str]): Danh sách các chuỗi text cần tạo embedding.

        Returns:
            List[List[float]]: Danh sách các vector embedding.
        """
        try:
            embeddings_np = asyncio.run(self.embed_texts(texts))
            return embeddings_np.tolist()
        except RuntimeError as e:
            if "cannot schedule new coroutine" in str(e):
                raise RuntimeError(
                    "Cannot call embed_documents in an async context. Use async_embed_texts instead."
                )
            raise
        except Exception as e:
            self.logger.error(f"Lỗi khi tạo embeddings cho documents: {str(e)}", exc_info=True)
            raise

    def embed_query(self, text: str) -> List[float]:
        """
        Tạo embedding cho một chuỗi query, trả về dạng list để tương thích với langchain.

        Args:
            text (str): Chuỗi query cần tạo embedding.

        Returns:
            List[float]: Vector embedding của query.
        """
        try:
            emb = asyncio.run(self.embed_text(text))
            return emb.tolist()
        except RuntimeError as e:
            if "cannot schedule new coroutine" in str(e):
                raise RuntimeError(
                    "Cannot call embed_query in an async context. Use async_embed_text instead."
                )
            raise
        except Exception as e:
            self.logger.error(f"Lỗi khi tạo embedding cho query: {str(e)}", exc_info=True)
            raise