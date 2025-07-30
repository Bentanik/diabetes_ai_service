from typing import List
import numpy as np
from core.llm.embedding_model import get_embedding_model
from utils import get_logger


class Embedding:
    def __init__(self):
        """
        Khởi tạo class Embedding để tạo embedding từ text hoặc list text.
        """
        self.logger = get_logger(__name__)
        self.model = get_embedding_model()

    def embed_text(self, text: str) -> np.ndarray:
        """
        Tạo embedding cho một chuỗi text.

        Args:
            text (str): Chuỗi text cần tạo embedding.

        Returns:
            np.ndarray: Vector embedding của text.
        """
        try:
            embedding = self.model.encode([text], show_progress_bar=False)[0]
            return embedding
        except Exception as e:
            self.logger.error(f"Lỗi khi tạo embedding cho text: {str(e)}")
            raise

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Tạo embedding cho danh sách các chuỗi text.

        Args:
            texts (List[str]): Danh sách các chuỗi text cần tạo embedding.

        Returns:
            np.ndarray: Ma trận embedding của các text (shape: [len(texts), vector_size]).
        """
        try:
            embeddings = self.model.encode(texts, show_progress_bar=False)
            return embeddings
        except Exception as e:
            self.logger.error(f"Lỗi khi tạo embedding cho list text: {str(e)}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings_np = self.embed_texts(texts)
        return embeddings_np.tolist()

    def embed_query(self, text: str) -> List[float]:
        emb = self.embed_text(text)
        return emb.tolist()


if __name__ == "__main__":
    embedding = Embedding()
