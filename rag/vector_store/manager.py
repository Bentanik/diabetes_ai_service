from typing import Dict, List
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from core.embedding import EmbeddingModel
import logging


class VectorStoreManager:
    def __init__(self, force_recreate: bool = False):
        self.logger = logging.getLogger(__name__)
        self.embedding_model = None
        self.client = QdrantClient(host="localhost", port=6333)
        self.vector_stores: Dict[str, Qdrant] = {}
        self.force_recreate = force_recreate

    async def create_collection_if_not_exists(
        self,
        collection_name: str,
        vector_size: int = 768,
        distance: Distance = Distance.COSINE,
    ) -> None:
        try:
            if self.embedding_model is None:
                embedding_model = await EmbeddingModel.get_instance()
                self.embedding_model = embedding_model.model

            if self.force_recreate and self.client.collection_exists(collection_name):
                self.logger.info(
                    f"Force recreate enabled. Deleting collection: {collection_name}"
                )
                self.client.delete_collection(collection_name)
                self.vector_stores.pop(collection_name, None)

            if not self.client.collection_exists(collection_name):
                self.logger.info(f"Creating new collection: {collection_name}")
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=distance),
                )
            else:
                self.logger.info(f"Collection '{collection_name}' already exists")

            # Initialize Qdrant LangChain wrapper if not already initialized
            if collection_name not in self.vector_stores:
                self.vector_stores[collection_name] = Qdrant(
                    client=self.client,
                    collection_name=collection_name,
                    embeddings=self.embedding_model,  # Use LangChain Embeddings
                )
                self.logger.info(
                    f"Initialized Qdrant vector store for collection: {collection_name}"
                )
        except Exception as e:
            self.logger.error(
                f"Failed to create/initialize collection '{collection_name}': {str(e)}",
                exc_info=True,
            )
            raise

    async def get_store(self, collection_name: str, vector_size: int = 768) -> Qdrant:
        try:
            if collection_name not in self.vector_stores:
                self.logger.info(
                    f"Collection '{collection_name}' not initialized, creating now."
                )
                await self.create_collection_if_not_exists(collection_name, vector_size)
            return self.vector_stores[collection_name]
        except Exception as e:
            self.logger.error(
                f"Failed to get vector store for collection '{collection_name}': {str(e)}",
                exc_info=True,
            )
            raise

    def get_collection_names(self) -> List[str]:
        try:
            return [col.name for col in self.client.get_collections().collections]
        except Exception as e:
            self.logger.error(
                f"Failed to get collection names: {str(e)}", exc_info=True
            )
            raise

    def delete_collection(self, collection_name: str) -> None:
        try:
            if self.client.collection_exists(collection_name):
                self.logger.info(f"Deleting collection: {collection_name}")
                self.client.delete_collection(collection_name)

            if collection_name in self.vector_stores:
                del self.vector_stores[collection_name]

            self.logger.info(f"Deleted collection: {collection_name}")
        except Exception as e:
            self.logger.error(
                f"Error deleting collection '{collection_name}': {str(e)}",
                exc_info=True,
            )
            raise
