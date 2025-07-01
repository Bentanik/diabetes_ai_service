"""
Dịch vụ Vector Store

Tích hợp Qdrant + LangChain
cho pipeline RAG tiếng Việt.

Tính năng:
- Cơ sở dữ liệu vector Qdrant tích hợp LangChain
- Tìm kiếm kết hợp (vector + BM25)
- Quản lý collection
- Lọc theo metadata
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import uuid
from datetime import datetime

from qdrant_client.models import FilterSelector, PointIdsList

try:
    from langchain_qdrant import QdrantVectorStore
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from qdrant_client import QdrantClient, AsyncQdrantClient
    from qdrant_client.models import (
        Distance,
        VectorParams,
        PointStruct,
        Filter,
        FieldCondition,
        MatchValue,
        SearchRequest,
        Batch,
        PayloadSchemaType,
        ExtendedPointId,
    )
    from qdrant_client.http import models
except ImportError as e:
    logging.error(f"Missing Qdrant dependencies: {e}")
    raise

logger = logging.getLogger(__name__)


@dataclass
class VectorStoreConfig:
    """Cấu hình cho vector store"""

    # Qdrant connection
    qdrant_url: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: Optional[str] = None

    # Collection settings
    collection_name: str = "vietnamese_documents"
    vector_size: int = 768  # E5-base dimension
    distance_metric: Distance = Distance.COSINE

    # Search settings
    search_limit: int = 10
    search_score_threshold: float = 0.7

    # Indexing settings
    batch_size: int = 100  # RagFlow-style batching
    enable_payload_indexing: bool = True


class QdrantVectorService:
    """
    Enhanced Qdrant vector service

    Kết hợp LangChain với RagFlow patterns cho performance tối ưu.
    """

    def __init__(
        self, embeddings: Embeddings, config: Optional[VectorStoreConfig] = None
    ):
        self.embeddings = embeddings
        self.config = config or VectorStoreConfig()

        # Initialize clients with proper typing
        self.client: Optional[QdrantClient] = None
        self.async_client: Optional[AsyncQdrantClient] = None
        self.vector_store: Optional[QdrantVectorStore] = None

        # Stats tracking như RagFlow
        self._stats = {
            "total_documents_indexed": 0,
            "total_searches_performed": 0,
            "total_indexing_time": 0.0,
            "total_search_time": 0.0,
            "last_operation_time": None,
        }

        self._initialize_clients()

    def _initialize_clients(self):
        """Initialize Qdrant clients"""
        try:
            # Sync client
            self.client = QdrantClient(
                host=self.config.qdrant_url,
                port=self.config.qdrant_port,
                api_key=self.config.qdrant_api_key,
            )

            # Async client
            self.async_client = AsyncQdrantClient(
                host=self.config.qdrant_url,
                port=self.config.qdrant_port,
                api_key=self.config.qdrant_api_key,
            )

            # LangChain vector store - fixed parameter name
            self.vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.config.collection_name,
                embedding=self.embeddings,  # Fixed: 'embedding' not 'embeddings'
            )

            logger.info(
                f"Qdrant clients initialized: {self.config.qdrant_url}:{self.config.qdrant_port}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize Qdrant clients: {e}")
            raise

    def create_collection(self, force_recreate: bool = False) -> bool:
        """
        Create Qdrant collection với optimization

        RagFlow pattern: auto-detect vector size và create optimized index
        """
        try:
            if self.client is None:
                logger.error("Qdrant client is not initialized.")
                return False

            collection_name = self.config.collection_name

            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_exists = any(
                getattr(col, "name", None) == collection_name for col in collections
            )

            if collection_exists:
                if force_recreate:
                    logger.info(f"Deleting existing collection: {collection_name}")
                    self.client.delete_collection(collection_name=collection_name)
                else:
                    logger.info(f"Collection {collection_name} already exists")
                    return True

            # Get embedding dimension safely
            vector_size = self.config.vector_size

            # Try to get actual embedding dimension if available
            if hasattr(self.embeddings, "__dict__"):
                # Check common attribute names used by embedding models
                for attr_name in [
                    "embedding_dimension",
                    "model_output_dim",
                    "vector_size",
                ]:
                    if hasattr(self.embeddings, attr_name):
                        attr_value = getattr(self.embeddings, attr_name)
                        if callable(attr_value):
                            try:
                                result = attr_value()
                                if isinstance(result, (int, str)):
                                    vector_size = int(result)
                                    break
                            except:
                                continue
                        elif isinstance(attr_value, (int, str)):
                            vector_size = int(attr_value)
                            break

            logger.info(f"Creating collection: {collection_name} (dim={vector_size})")

            # Ensure vector_size is int
            vector_size = int(vector_size)

            # Create collection with optimized settings
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=self.config.distance_metric,
                ),
            )

            # Create payload indices for filtering như RagFlow
            if self.config.enable_payload_indexing:
                self._create_payload_indices(collection_name)

            logger.info(f"Collection {collection_name} created successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return False

    def _create_payload_indices(self, collection_name: str):
        """Create payload indices for efficient filtering"""
        try:
            if self.client is None:
                logger.error("Qdrant client is not initialized.")
                return

            # Index common metadata fields
            index_fields = [
                "source_file",
                "doc_id",
                "chunk_id",
                "embedding_model",
                "file_extension",
                "processing_timestamp",
            ]

            for field in index_fields:
                try:
                    # Use proper PayloadSchemaType enum
                    self.client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field,
                        field_schema=PayloadSchemaType.KEYWORD,
                    )
                    logger.debug(f"Created index for field: {field}")
                except Exception as e:
                    logger.debug(f"Failed to create index for {field}: {e}")

        except Exception as e:
            logger.warning(f"Failed to create payload indices: {e}")

    def add_documents(
        self, documents: List[Document], batch_size: Optional[int] = None
    ) -> List[str]:
        """
        Add documents to vector store với RagFlow batch processing
        """
        if not documents:
            return []

        if self.vector_store is None:
            logger.error("Vector store is not initialized.")
            return []

        start_time = datetime.now()
        batch_size = batch_size or self.config.batch_size

        try:
            # Ensure collection exists
            self.create_collection()

            # Process in batches
            all_ids = []
            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]

                # Generate IDs for batch
                batch_ids = [str(uuid.uuid4()) for _ in batch]

                # Add batch to vector store
                self.vector_store.add_documents(documents=batch, ids=batch_ids)

                all_ids.extend(batch_ids)
                logger.debug(
                    f"Indexed batch {i//batch_size + 1}: {len(batch)} documents"
                )

            # Update stats
            indexing_time = (datetime.now() - start_time).total_seconds()
            self._stats["total_documents_indexed"] += len(documents)
            self._stats["total_indexing_time"] += indexing_time
            self._stats["last_operation_time"] = datetime.now().isoformat()

            logger.info(f"Indexed {len(documents)} documents in {indexing_time:.2f}s")
            return all_ids

        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise

    async def aadd_documents(
        self, documents: List[Document], batch_size: Optional[int] = None
    ) -> List[str]:
        """Async version of add_documents"""
        # For now, run sync version in executor
        # Todo: Implement true async batch processing
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.add_documents, documents, batch_size
        )

    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Similarity search với filtering support
        """
        if self.vector_store is None:
            logger.error("Vector store is not initialized.")
            return []

        k = k or self.config.search_limit
        score_threshold = score_threshold or self.config.search_score_threshold

        start_time = datetime.now()

        try:
            # Build filter từ conditions
            search_filter = (
                self._build_filter(filter_conditions) if filter_conditions else None
            )

            # Perform search
            results = self.vector_store.similarity_search(
                query=query, k=k, filter=search_filter
            )

            # Update stats
            search_time = (datetime.now() - start_time).total_seconds()
            self._stats["total_searches_performed"] += 1
            self._stats["total_search_time"] += search_time

            logger.debug(
                f"Search completed: {len(results)} results in {search_time:.3f}s"
            )
            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def similarity_search_with_score(
        self,
        query: str,
        k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Similarity search with scores - RagFlow pattern
        """
        if self.vector_store is None:
            logger.error("Vector store is not initialized.")
            return []

        k = k or self.config.search_limit
        score_threshold = score_threshold or self.config.search_score_threshold

        start_time = datetime.now()

        try:
            search_filter = (
                self._build_filter(filter_conditions) if filter_conditions else None
            )

            results = self.vector_store.similarity_search_with_score(
                query=query, k=k, filter=search_filter, score_threshold=score_threshold
            )

            # Update stats
            search_time = (datetime.now() - start_time).total_seconds()
            self._stats["total_searches_performed"] += 1
            self._stats["total_search_time"] += search_time
            self._stats["last_operation_time"] = datetime.now().isoformat()

            logger.debug(
                f"Search with scores: {len(results)} results in {search_time:.3f}s"
            )
            return results

        except Exception as e:
            logger.error(f"Search with score failed: {e}")
            return []

    def _build_filter(self, conditions: Dict[str, Any]) -> Optional[Filter]:
        """
        Build Qdrant filter from conditions

        RagFlow-style filtering support
        """
        field_conditions = []

        for field, value in conditions.items():
            if isinstance(value, (str, int, bool)):
                # Handle different value types properly
                field_conditions.append(
                    FieldCondition(key=field, match=MatchValue(value=value))
                )
            elif isinstance(value, list):
                # Multiple values - create separate conditions for OR logic
                for v in value:
                    if isinstance(v, (str, int, bool)):
                        field_conditions.append(
                            FieldCondition(key=field, match=MatchValue(value=v))
                        )

        return Filter(must=field_conditions) if field_conditions else None

    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information"""
        try:
            if self.client is None:
                logger.error("Qdrant client is not initialized.")
                return {}

            collection_info = self.client.get_collection(self.config.collection_name)

            # Simplified collection info - use config values if Qdrant API structure is complex
            try:
                vectors_config = collection_info.config.params.vectors
                if isinstance(vectors_config, dict):
                    # Multiple vector configs
                    first_key = next(iter(vectors_config))
                    first_vector = vectors_config[first_key]
                    distance = first_vector.distance
                    size = first_vector.size
                else:
                    # Single vector config
                    distance = vectors_config.distance if vectors_config else None
                    size = vectors_config.size if vectors_config else None
            except:
                # Fallback to config values
                distance = self.config.distance_metric
                size = self.config.vector_size

            return {
                "name": self.config.collection_name,
                "vector_size": size,
                "distance_metric": distance,
                "points_count": collection_info.points_count,
                "indexed_count": collection_info.indexed_vectors_count,
                "status": collection_info.status,
            }

        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}

    def delete_documents(
        self,
        document_ids: Optional[List[Union[str, int]]] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Delete documents by IDs or filter conditions
        """
        try:
            if self.client is None:
                logger.error("Qdrant client is not initialized.")
                return False

            if document_ids:
                self.client.delete(
                    collection_name=self.config.collection_name,
                    points_selector=PointIdsList(points=document_ids),
                )
                logger.info(f"Deleted {len(document_ids)} documents by ID")

            elif filter_conditions:
                search_filter = self._build_filter(filter_conditions)
                if search_filter is not None:
                    self.client.delete(
                        collection_name=self.config.collection_name,
                        points_selector=FilterSelector(filter=search_filter),
                    )
                    logger.info("Deleted documents by filter")

            else:
                logger.warning("No document IDs or filter conditions provided.")
                return False

            return True

        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics - RagFlow monitoring pattern"""
        stats = self._stats.copy()

        # Add collection info
        collection_info = self.get_collection_info()
        stats.update(collection_info)

        # Calculate averages
        if stats["total_searches_performed"] > 0:
            stats["avg_search_time"] = (
                stats["total_search_time"] / stats["total_searches_performed"]
            )

        if stats["total_documents_indexed"] > 0:
            stats["avg_indexing_time_per_doc"] = (
                stats["total_indexing_time"] / stats["total_documents_indexed"]
            )

        return stats

    def reset_stats(self):
        """Reset statistics"""
        for key in self._stats:
            if key != "last_operation_time":
                self._stats[key] = 0


# Factory functions
def create_qdrant_vector_store(
    embeddings: Embeddings, config: Optional[VectorStoreConfig] = None
) -> QdrantVectorService:
    """Create Qdrant vector store service"""
    return QdrantVectorService(embeddings, config)


def create_vietnamese_vector_store(
    embeddings: Embeddings, collection_name: str = "vietnamese_docs"
) -> QdrantVectorService:
    """Create vector store optimized for Vietnamese documents"""
    config = VectorStoreConfig(
        collection_name=collection_name,
        vector_size=768,  # E5-base
        distance_metric=Distance.COSINE,
        search_limit=10,
        search_score_threshold=0.7,
        batch_size=100,
        enable_payload_indexing=True,
    )
    return QdrantVectorService(embeddings, config)
