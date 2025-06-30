"""
Enhanced Retrieval Service

Kết hợp RagFlow retrieval patterns với LangChain cho Vietnamese RAG pipeline.
Hỗ trợ hybrid search (vector + BM25), reranking, và filtering tối ưu.

Features:
- Vector similarity search
- BM25 text search (hybrid approach)
- Score combination & reranking
- Metadata filtering
- RagFlow-style result processing
- Async operations support
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import time
from datetime import datetime

try:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.retrievers import BaseRetriever
    from langchain.retrievers import BM25Retriever, EnsembleRetriever
    from langchain_community.retrievers import QdrantSparseVectorRetriever
except ImportError as e:
    logging.error(f"Missing LangChain dependencies: {e}")
    raise

from .vector_store import QdrantVectorService
from .embedding import EmbeddingService

logger = logging.getLogger(__name__)


@dataclass
class RetrievalConfig:
    """Cấu hình cho retrieval service"""

    # Search parameters
    k: int = 10  # Top-K results
    score_threshold: float = 0.7  # Minimum similarity score
    max_results: int = 50  # Maximum results to consider

    # Hybrid search weights - RagFlow pattern
    vector_weight: float = 0.7  # Vector similarity weight
    bm25_weight: float = 0.3  # BM25 text weight

    # Reranking settings
    enable_reranking: bool = True
    rerank_top_k: int = 20  # Rerank top N results

    # Performance settings
    search_timeout: float = 30.0  # Search timeout in seconds
    enable_caching: bool = True


class HybridRetriever:
    """
    Hybrid retriever kết hợp vector search và BM25

    Inspired by RagFlow's hybrid approach với LangChain implementation.
    """

    def __init__(
        self,
        vector_service: QdrantVectorService,
        embedding_service: EmbeddingService,
        config: Optional[RetrievalConfig] = None,
    ):
        self.vector_service = vector_service
        self.embedding_service = embedding_service
        self.config = config or RetrievalConfig()

        # BM25 retriever for text search
        self.bm25_retriever = None
        self.ensemble_retriever = None

        # Document corpus for BM25
        self._document_corpus = []

        # Stats tracking
        self._stats = {
            "total_searches": 0,
            "total_search_time": 0.0,
            "avg_search_time": 0.0,
            "vector_search_time": 0.0,
            "bm25_search_time": 0.0,
            "rerank_time": 0.0,
            "last_search_time": None,
        }

        logger.info("HybridRetriever initialized")

    def update_document_corpus(self, documents: List[Document]):
        """
        Update document corpus for BM25 retriever

        RagFlow pattern: maintain text corpus for hybrid search
        """
        try:
            self._document_corpus = documents

            # Initialize BM25 retriever với document corpus
            if documents:
                self.bm25_retriever = BM25Retriever.from_documents(
                    documents=documents, k=self.config.k
                )

                # Create ensemble retriever combining vector + BM25
                if self.vector_service.vector_store is not None:
                    vector_retriever = self.vector_service.vector_store.as_retriever(
                        search_kwargs={"k": self.config.k}
                    )
                else:
                    logger.warning("Vector store not initialized")
                    vector_retriever = None

                self.ensemble_retriever = EnsembleRetriever(
                    retrievers=[vector_retriever, self.bm25_retriever],
                    weights=[self.config.vector_weight, self.config.bm25_weight],
                )

                logger.info(f"Updated BM25 corpus with {len(documents)} documents")

        except Exception as e:
            logger.error(f"Failed to update document corpus: {e}")

    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
        search_type: str = "hybrid",  # "vector", "bm25", "hybrid"
    ) -> List[Document]:
        """
        Perform similarity search với multiple strategies

        Args:
            query: Search query
            k: Number of results
            score_threshold: Minimum similarity score
            filter_conditions: Metadata filtering
            search_type: "vector", "bm25", or "hybrid"
        """
        start_time = time.time()
        k = k or self.config.k
        score_threshold = score_threshold or self.config.score_threshold

        try:
            results = []

            if search_type == "vector":
                results = self._vector_search(
                    query, k, score_threshold, filter_conditions
                )

            elif search_type == "bm25":
                results = self._bm25_search(query, k)

            elif search_type == "hybrid":
                results = self._hybrid_search(
                    query, k, score_threshold, filter_conditions
                )

            else:
                raise ValueError(f"Unknown search_type: {search_type}")

            # Apply reranking if enabled
            if self.config.enable_reranking and len(results) > 1:
                results = self._rerank_results(query, results)

            # Update stats
            search_time = time.time() - start_time
            self._update_search_stats(search_time)

            logger.info(
                f"{search_type} search: {len(results)} results in {search_time:.3f}s"
            )
            return results[:k]  # Return top-k

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
        k = k or self.config.k
        score_threshold = score_threshold or self.config.score_threshold

        try:
            # Use vector search with scores
            results = self.vector_service.similarity_search_with_score(
                query=query,
                k=k,
                score_threshold=score_threshold,
                filter_conditions=filter_conditions,
            )

            logger.debug(f"Search with scores: {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Search with score failed: {e}")
            return []

    def _vector_search(
        self,
        query: str,
        k: int,
        score_threshold: float,
        filter_conditions: Optional[Dict[str, Any]],
    ) -> List[Document]:
        """Vector similarity search"""
        vector_start = time.time()

        results = self.vector_service.similarity_search(
            query=query,
            k=k,
            score_threshold=score_threshold,
            filter_conditions=filter_conditions,
        )

        self._stats["vector_search_time"] += time.time() - vector_start
        return results

    def _bm25_search(self, query: str, k: int) -> List[Document]:
        """BM25 text search"""
        if not self.bm25_retriever:
            logger.warning("BM25 retriever not initialized")
            return []

        bm25_start = time.time()

        try:
            # Use BM25 retriever
            results = self.bm25_retriever.get_relevant_documents(query)

            self._stats["bm25_search_time"] += time.time() - bm25_start
            return results[:k]

        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []

    def _hybrid_search(
        self,
        query: str,
        k: int,
        score_threshold: float,
        filter_conditions: Optional[Dict[str, Any]],
    ) -> List[Document]:
        """
        Hybrid search combining vector + BM25

        RagFlow-inspired score combination
        """
        if not self.ensemble_retriever:
            logger.warning(
                "Ensemble retriever not available, falling back to vector search"
            )
            return self._vector_search(query, k, score_threshold, filter_conditions)

        try:
            # Use ensemble retriever for hybrid search
            results = self.ensemble_retriever.get_relevant_documents(query)

            # Apply score threshold filtering for vector component
            # Note: Ensemble retriever doesn't directly support score filtering
            # So we do post-processing
            filtered_results = []

            for doc in results:
                # For hybrid results, we trust the ensemble scoring
                # Or optionally re-score with vector similarity
                filtered_results.append(doc)

            return filtered_results[:k]

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            # Fallback to vector search
            return self._vector_search(query, k, score_threshold, filter_conditions)

    def _rerank_results(self, query: str, results: List[Document]) -> List[Document]:
        """
        Rerank results based on query relevance

        Simple implementation - can be enhanced with dedicated reranking models
        """
        rerank_start = time.time()

        try:
            # For now, use simple text similarity for reranking
            # In production, could use cross-encoder models like RagFlow

            scored_results = []
            query_lower = query.lower()

            for doc in results[: self.config.rerank_top_k]:
                content_lower = doc.page_content.lower()

                # Simple keyword overlap score
                query_words = set(query_lower.split())
                content_words = set(content_lower.split())
                overlap = len(query_words.intersection(content_words))
                overlap_score = overlap / max(len(query_words), 1)

                scored_results.append((doc, overlap_score))

            # Sort by rerank score
            scored_results.sort(key=lambda x: x[1], reverse=True)
            reranked = [doc for doc, score in scored_results]

            # Add remaining results without reranking
            reranked.extend(results[self.config.rerank_top_k :])

            self._stats["rerank_time"] += time.time() - rerank_start
            return reranked

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return results

    def _update_search_stats(self, search_time: float):
        """Update search statistics"""
        self._stats["total_searches"] += 1
        self._stats["total_search_time"] += search_time
        self._stats["avg_search_time"] = (
            self._stats["total_search_time"] / self._stats["total_searches"]
        )
        self._stats["last_search_time"] = datetime.now().isoformat()

    async def asimilarity_search(
        self,
        query: str,
        k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
        search_type: str = "hybrid",
    ) -> List[Document]:
        """Async version of similarity search"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.similarity_search,
            query,
            k,
            score_threshold,
            filter_conditions,
            search_type,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics"""
        return self._stats.copy()

    def reset_stats(self):
        """Reset statistics"""
        for key in self._stats:
            if key != "last_search_time":
                self._stats[key] = 0.0


class RetrievalService:
    """
    High-level retrieval service

    Orchestrates embedding, vector storage, and retrieval operations.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_service: QdrantVectorService,
        config: Optional[RetrievalConfig] = None,
    ):
        self.embedding_service = embedding_service
        self.vector_service = vector_service
        self.config = config or RetrievalConfig()

        # Initialize hybrid retriever
        self.hybrid_retriever = HybridRetriever(
            vector_service=vector_service,
            embedding_service=embedding_service,
            config=config,
        )

        logger.info("RetrievalService initialized successfully")

    def index_documents(
        self, documents: List[Document], batch_size: Optional[int] = None
    ) -> List[str]:
        """
        Index documents: embed + store in vector database

        Complete pipeline từ documents đến searchable index
        """
        if not documents:
            return []

        try:
            logger.info(f"Indexing {len(documents)} documents...")

            # Step 1: Generate embeddings
            embedded_docs = self.embedding_service.embed_documents_from_chunks(
                documents
            )

            # Step 2: Store in vector database
            doc_ids = self.vector_service.add_documents(embedded_docs, batch_size)

            # Step 3: Update BM25 corpus for hybrid search
            self.hybrid_retriever.update_document_corpus(embedded_docs)

            logger.info(f"Successfully indexed {len(documents)} documents")
            return doc_ids

        except Exception as e:
            logger.error(f"Failed to index documents: {e}")
            raise

    async def aindex_documents(
        self, documents: List[Document], batch_size: Optional[int] = None
    ) -> List[str]:
        """Async version of document indexing"""
        try:
            # Async embedding
            embedded_docs = await self.embedding_service.aembed_documents_from_chunks(
                documents
            )

            # Sync vector store (for now)
            doc_ids = self.vector_service.add_documents(embedded_docs, batch_size)

            # Update BM25 corpus
            self.hybrid_retriever.update_document_corpus(embedded_docs)

            return doc_ids

        except Exception as e:
            logger.error(f"Failed to async index documents: {e}")
            raise

    def search(
        self,
        query: str,
        k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
        search_type: str = "hybrid",
    ) -> List[Document]:
        """
        Search documents using hybrid retrieval
        """
        return self.hybrid_retriever.similarity_search(
            query=query,
            k=k,
            score_threshold=score_threshold,
            filter_conditions=filter_conditions,
            search_type=search_type,
        )

    async def asearch(
        self,
        query: str,
        k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
        search_type: str = "hybrid",
    ) -> List[Document]:
        """Async search"""
        return await self.hybrid_retriever.asimilarity_search(
            query=query,
            k=k,
            score_threshold=score_threshold,
            filter_conditions=filter_conditions,
            search_type=search_type,
        )

    def search_with_scores(
        self,
        query: str,
        k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """Search with similarity scores"""
        return self.hybrid_retriever.similarity_search_with_score(
            query=query,
            k=k,
            score_threshold=score_threshold,
            filter_conditions=filter_conditions,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        stats = {
            "embedding_stats": self.embedding_service.get_stats(),
            "vector_store_stats": self.vector_service.get_stats(),
            "retrieval_stats": self.hybrid_retriever.get_stats(),
        }
        return stats


# Factory functions
def create_retrieval_service(
    embedding_service: EmbeddingService,
    vector_service: QdrantVectorService,
    config: Optional[RetrievalConfig] = None,
) -> RetrievalService:
    """Create retrieval service instance"""
    return RetrievalService(embedding_service, vector_service, config)


def create_vietnamese_retrieval_service(
    embedding_service: EmbeddingService, vector_service: QdrantVectorService
) -> RetrievalService:
    """Create retrieval service optimized for Vietnamese"""
    config = RetrievalConfig(
        k=10,
        score_threshold=0.7,
        vector_weight=0.7,  # Favor vector search for Vietnamese
        bm25_weight=0.3,
        enable_reranking=True,
        rerank_top_k=20,
    )
    return RetrievalService(embedding_service, vector_service, config)
