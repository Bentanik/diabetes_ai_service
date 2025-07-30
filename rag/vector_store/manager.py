from typing import Dict
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from rag.embedding import Embedding
from utils import get_logger


class VectorStoreManager:
    def __init__(self, force_recreate: bool = False):
        self.logger = get_logger(__name__)
        self.embedding = Embedding()
        self.client = QdrantClient(host="localhost", port=6333)
        self.vector_stores: Dict[str, Qdrant] = {}
        self.force_recreate = force_recreate

    def create_collection_if_not_exists(
        self,
        collection_name: str,
        vector_size: int,
        distance: Distance = Distance.COSINE,
    ) -> None:
        if self.force_recreate and self.client.collection_exists(collection_name):
            self.logger.info(f"Xóa collection cũ: {collection_name}")
            self.client.delete_collection(collection_name)
            if collection_name in self.vector_stores:
                del self.vector_stores[collection_name]

        if not self.client.collection_exists(collection_name):
            self.logger.info(f"Tạo collection mới: {collection_name}")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=distance),
            )
        else:
            self.logger.info(f"Collection đã tồn tại: {collection_name}")

        if collection_name not in self.vector_stores:
            self.vector_stores[collection_name] = Qdrant(
                client=self.client,
                collection_name=collection_name,
                embeddings=self.embedding,
            )
            self.logger.info(f"Khởi tạo Qdrant vector store cho {collection_name}")

    def get_store(self, collection_name: str, vector_size: int = 768) -> Qdrant:
        if collection_name not in self.vector_stores:
            self.logger.info(f"Collection {collection_name} chưa có, tạo mới")
            self.create_collection_if_not_exists(collection_name, vector_size)
        return self.vector_stores[collection_name]

    def get_collection_names(self) -> list[str]:
        return list(self.vector_stores.keys())
