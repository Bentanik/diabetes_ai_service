"""
LangChain Retriever tích hợp với RAG Service

Kết hợp:
- LangChain BaseRetriever interface
- RAG Embedding Service (E5 Multilingual)
- Qdrant Vector Store
"""

import logging
from typing import List, Optional, Any, Dict
from dataclasses import dataclass
from langchain.schema import BaseRetriever, Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun

from aiservice.src.rag.embedding import (
    EmbeddingService,
    create_vietnamese_optimized_embeddings,
)
from aiservice.src.rag.vector_store import (
    QdrantVectorService,
    create_vietnamese_vector_store,
)

logger = logging.getLogger(__name__)


@dataclass
class RetrieverConfig:
    """Configuration for the Retriever."""

    top_k: int = 4
    score_threshold: float = 0.7


class Retriever(BaseRetriever):
    """LangChain retriever optimized for Vietnamese using RAG components."""

    def __init__(
        self,
        *,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[QdrantVectorService] = None,
        top_k: int = 4,
        score_threshold: float = 0.7,
    ):
        """Initialize retriever with RAG components.

        Args:
            embedding_service: Service for creating embeddings (uses Vietnamese optimized if None)
            vector_store: Vector store service (creates new one if None)
            top_k: Number of documents to retrieve
            score_threshold: Minimum similarity score threshold
        """
        super().__init__()

        # Initialize embedding service
        self._embedding_service = (
            embedding_service or create_vietnamese_optimized_embeddings()
        )

        # Initialize vector store if not provided
        if vector_store is None:
            vector_store = create_vietnamese_vector_store(
                embeddings=self._embedding_service.get_langchain_embeddings()
            )
        self._vector_store = vector_store

        # Create config
        self._config = RetrieverConfig(top_k=top_k, score_threshold=score_threshold)

        logger.info("Vietnamese retriever initialized successfully")

    @property
    def embedding_service(self) -> EmbeddingService:
        """Get the embedding service."""
        return self._embedding_service

    @property
    def vector_store(self) -> QdrantVectorService:
        """Get the vector store."""
        return self._vector_store

    @property
    def config(self) -> RetrieverConfig:
        """Get the configuration."""
        return self._config

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant to query.

        Args:
            query: Query string
            run_manager: Callback manager

        Returns:
            List of relevant documents
        """
        try:
            if not self.vector_store:
                raise ValueError("Vector store is not initialized")

            # Get documents with scores
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=self.config.top_k,
                score_threshold=self.config.score_threshold,
            )

            # Extract documents and add scores to metadata
            documents = []
            for doc, score in results:
                if not isinstance(doc, Document):
                    logger.warning(f"Unexpected document type: {type(doc)}")
                    continue

                # Create a new document with score in metadata
                new_doc = Document(
                    page_content=doc.page_content,
                    metadata={**(doc.metadata or {}), "score": score},
                )
                documents.append(new_doc)

            return documents

        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            raise

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Async version of document retrieval."""
        # For now, just call the sync version
        return self._get_relevant_documents(query, run_manager=run_manager)
