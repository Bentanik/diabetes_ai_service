"""
Complete RAG Service Integration

Tích hợp hoàn chỉnh embedding + vector store + retrieval 
cho Vietnamese RAG pipeline với RagFlow patterns.

Features:
- End-to-end document processing
- Embedding generation với multilingual-e5-base
- Qdrant vector storage
- Hybrid search (vector + BM25)
- Performance monitoring
- Async operations
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

try:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain.retrievers import BM25Retriever, EnsembleRetriever
except ImportError as e:
    logging.error(f"Missing LangChain dependencies: {e}")
    raise

from .embedding import (
    EmbeddingService,
    EmbeddingConfig,
    create_vietnamese_optimized_embeddings,
)
from .vector_store import (
    QdrantVectorService,
    VectorStoreConfig,
    create_vietnamese_vector_store,
)
from .chunking import Chunking, ChunkingConfig

logger = logging.getLogger(__name__)


@dataclass
class RAGServiceConfig:
    """Cấu hình tổng thể cho RAG service"""

    # Embedding settings
    embedding_model: str = "intfloat/multilingual-e5-base"
    embedding_device: str = "auto"

    # Vector store settings
    collection_name: str = "vietnamese_rag_collection"
    qdrant_url: str = "localhost"
    qdrant_port: int = 6333

    # Search settings
    default_k: int = 10
    score_threshold: float = 0.7
    vector_weight: float = 0.7
    bm25_weight: float = 0.3

    # Performance settings
    batch_size: int = 100
    enable_hybrid_search: bool = True
    enable_reranking: bool = True


class VietnameseRAGService:
    """
    Complete Vietnamese RAG service

    Kết hợp tất cả components: chunking, embedding, vector store, retrieval
    """

    def __init__(self, config: Optional[RAGServiceConfig] = None):
        self.config = config or RAGServiceConfig()

        # Components
        self.chunking = None
        self.embedding_service = None
        self.vector_service = None
        self.bm25_retriever = None
        self.ensemble_retriever = None

        # Document corpus for BM25
        self._document_corpus = []

        # Service stats
        self._stats = {
            "total_documents_processed": 0,
            "total_chunks_created": 0,
            "total_embeddings_generated": 0,
            "total_documents_indexed": 0,
            "total_searches_performed": 0,
            "service_initialized_at": datetime.now().isoformat(),
            "last_operation_time": None,
        }

        self._initialize_services()

    def _initialize_services(self):
        """Initialize all RAG components"""
        try:
            logger.info("Initializing Vietnamese RAG Service...")

            # 1. Initialize chunking
            chunking_config = ChunkingConfig(
                chunk_size=512, chunk_overlap=64, min_chunk_size=50, max_chunk_size=1024
            )
            self.chunking = Chunking(chunking_config)
            logger.info("✓ Chunking service initialized")

            # 2. Initialize embedding service
            embedding_config = EmbeddingConfig(
                model_name=self.config.embedding_model,
                device=self.config.embedding_device,
                batch_size=16,
                max_tokens=512,
                normalize_embeddings=True,
                query_instruction="query: ",
                passage_instruction="passage: ",
            )
            self.embedding_service = EmbeddingService(embedding_config)
            logger.info("✓ Embedding service initialized")

            # 3. Initialize vector store
            vector_config = VectorStoreConfig(
                collection_name=self.config.collection_name,
                qdrant_url=self.config.qdrant_url,
                qdrant_port=self.config.qdrant_port,
                vector_size=self.embedding_service.get_embedding_dimension(),
                batch_size=self.config.batch_size,
                enable_payload_indexing=True,
            )

            embeddings = self.embedding_service.get_langchain_embeddings()
            self.vector_service = QdrantVectorService(embeddings, vector_config)
            logger.info("✓ Vector store service initialized")

            # 4. Create collection if needed
            self.vector_service.create_collection()
            logger.info("✓ Vector collection ready")

            logger.info("🎉 Vietnamese RAG Service initialized successfully!")

        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {e}")
            raise

    def process_and_index_documents(
        self,
        documents: List[Document],
        chunk_documents: bool = True,
        batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Complete pipeline: documents → chunks → embeddings → vector store

        Args:
            documents: Raw documents to process
            chunk_documents: Whether to chunk the documents
            batch_size: Batch size for indexing

        Returns:
            Dict with processing results and stats
        """
        start_time = datetime.now()
        batch_size = batch_size or self.config.batch_size

        try:
            logger.info(f"Processing {len(documents)} documents...")

            # Step 1: Chunk documents if needed
            if chunk_documents:
                chunks = self.chunking.chunk_documents(documents)
                logger.info(
                    f"Created {len(chunks)} chunks from {len(documents)} documents"
                )
            else:
                chunks = documents
                logger.info(f"Using {len(documents)} documents as-is (no chunking)")

            # Step 2: Generate embeddings và add to metadata
            embedded_chunks = self.embedding_service.embed_documents_from_chunks(chunks)
            logger.info(f"Generated embeddings for {len(embedded_chunks)} chunks")

            # Step 3: Index to vector store
            doc_ids = self.vector_service.add_documents(embedded_chunks, batch_size)
            logger.info(f"Indexed {len(doc_ids)} documents to vector store")

            # Step 4: Update BM25 corpus for hybrid search
            if self.config.enable_hybrid_search:
                self._update_bm25_corpus(embedded_chunks)
                logger.info("Updated BM25 corpus for hybrid search")

            # Update stats
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_processing_stats(
                len(documents), len(chunks), len(embedded_chunks)
            )

            result = {
                "success": True,
                "documents_processed": len(documents),
                "chunks_created": len(chunks) if chunk_documents else 0,
                "embeddings_generated": len(embedded_chunks),
                "documents_indexed": len(doc_ids),
                "processing_time_seconds": processing_time,
                "document_ids": (
                    doc_ids[:10] if len(doc_ids) > 10 else doc_ids
                ),  # Sample IDs
                "collection_info": self.vector_service.get_collection_info(),
            }

            logger.info(f"✅ Processing completed in {processing_time:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Failed to process and index documents: {e}")
            return {
                "success": False,
                "error": str(e),
                "documents_processed": 0,
                "processing_time_seconds": (
                    datetime.now() - start_time
                ).total_seconds(),
            }

    def search(
        self,
        query: str,
        k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
        search_type: str = "hybrid",  # "vector", "bm25", "hybrid"
    ) -> List[Document]:
        """
        Search documents using selected strategy

        Args:
            query: Search query
            k: Number of results to return
            score_threshold: Minimum similarity score
            filter_conditions: Metadata filtering conditions
            search_type: "vector", "bm25", or "hybrid"
        """
        start_time = datetime.now()
        k = k or self.config.default_k
        score_threshold = score_threshold or self.config.score_threshold

        try:
            logger.info(f"Searching with query: '{query}' (type: {search_type})")

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
            search_time = (datetime.now() - start_time).total_seconds()
            self._stats["total_searches_performed"] += 1
            self._stats["last_operation_time"] = datetime.now().isoformat()

            logger.info(
                f"Search completed: {len(results)} results in {search_time:.3f}s"
            )
            return results[:k]

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def search_with_scores(
        self,
        query: str,
        k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """Search with similarity scores"""
        k = k or self.config.default_k
        score_threshold = score_threshold or self.config.score_threshold

        try:
            results = self.vector_service.similarity_search_with_score(
                query=query,
                k=k,
                score_threshold=score_threshold,
                filter_conditions=filter_conditions,
            )

            self._stats["total_searches_performed"] += 1
            self._stats["last_operation_time"] = datetime.now().isoformat()

            return results

        except Exception as e:
            logger.error(f"Search with scores failed: {e}")
            return []

    def _vector_search(
        self,
        query: str,
        k: int,
        score_threshold: float,
        filter_conditions: Optional[Dict[str, Any]],
    ) -> List[Document]:
        """Vector similarity search"""
        return self.vector_service.similarity_search(
            query=query,
            k=k,
            score_threshold=score_threshold,
            filter_conditions=filter_conditions,
        )

    def _bm25_search(self, query: str, k: int) -> List[Document]:
        """BM25 text search"""
        if not self.bm25_retriever:
            logger.warning("BM25 retriever not initialized")
            return []

        try:
            results = self.bm25_retriever.get_relevant_documents(query)
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
        """Hybrid search combining vector + BM25"""
        if not self.ensemble_retriever:
            logger.warning(
                "Ensemble retriever not available, falling back to vector search"
            )
            return self._vector_search(query, k, score_threshold, filter_conditions)

        try:
            results = self.ensemble_retriever.get_relevant_documents(query)
            return results[:k]
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return self._vector_search(query, k, score_threshold, filter_conditions)

    def _update_bm25_corpus(self, documents: List[Document]):
        """Update BM25 retriever corpus"""
        try:
            self._document_corpus.extend(documents)

            # Recreate BM25 retriever with updated corpus
            if self._document_corpus:
                self.bm25_retriever = BM25Retriever.from_documents(
                    documents=self._document_corpus, k=self.config.default_k
                )

                # Create ensemble retriever
                if self.vector_service.vector_store:
                    vector_retriever = self.vector_service.vector_store.as_retriever(
                        search_kwargs={"k": self.config.default_k}
                    )

                    self.ensemble_retriever = EnsembleRetriever(
                        retrievers=[vector_retriever, self.bm25_retriever],
                        weights=[self.config.vector_weight, self.config.bm25_weight],
                    )

                logger.debug(
                    f"Updated BM25 corpus: {len(self._document_corpus)} total documents"
                )

        except Exception as e:
            logger.error(f"Failed to update BM25 corpus: {e}")

    def _rerank_results(self, query: str, results: List[Document]) -> List[Document]:
        """Simple reranking based on keyword overlap"""
        try:
            scored_results = []
            query_lower = query.lower()
            query_words = set(query_lower.split())

            for doc in results[:20]:  # Rerank top 20
                content_lower = doc.page_content.lower()
                content_words = set(content_lower.split())
                overlap = len(query_words.intersection(content_words))
                overlap_score = overlap / max(len(query_words), 1)
                scored_results.append((doc, overlap_score))

            # Sort by rerank score
            scored_results.sort(key=lambda x: x[1], reverse=True)
            reranked = [doc for doc, score in scored_results]

            # Add remaining results
            reranked.extend(results[20:])

            return reranked

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return results

    def _update_processing_stats(
        self, num_docs: int, num_chunks: int, num_embeddings: int
    ):
        """Update processing statistics"""
        self._stats["total_documents_processed"] += num_docs
        self._stats["total_chunks_created"] += num_chunks
        self._stats["total_embeddings_generated"] += num_embeddings
        self._stats["total_documents_indexed"] += num_embeddings
        self._stats["last_operation_time"] = datetime.now().isoformat()

    def get_service_stats(self) -> Dict[str, Any]:
        """Get comprehensive service statistics"""
        stats = self._stats.copy()

        # Add component stats
        stats["embedding_stats"] = self.embedding_service.get_stats()
        stats["vector_store_stats"] = self.vector_service.get_stats()
        stats["collection_info"] = self.vector_service.get_collection_info()

        return stats

    def export_stats_to_json(self, file_path: str):
        """Export statistics to JSON file"""
        try:
            stats = self.get_service_stats()
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            logger.info(f"Stats exported to {file_path}")
        except Exception as e:
            logger.error(f"Failed to export stats: {e}")


# Factory functions
def create_vietnamese_rag_service(
    collection_name: str = "vietnamese_documents",
    qdrant_url: str = "localhost",
    qdrant_port: int = 6333,
) -> VietnameseRAGService:
    """Create Vietnamese RAG service with default optimizations"""
    config = RAGServiceConfig(
        embedding_model="intfloat/multilingual-e5-base",
        collection_name=collection_name,
        qdrant_url=qdrant_url,
        qdrant_port=qdrant_port,
        default_k=10,
        score_threshold=0.7,
        vector_weight=0.7,  # Favor vector search for Vietnamese
        bm25_weight=0.3,
        enable_hybrid_search=True,
        enable_reranking=True,
    )
    return VietnameseRAGService(config)


def get_rag_service() -> VietnameseRAGService:
    """Get default RAG service instance - singleton pattern"""
    if not hasattr(get_rag_service, "_instance"):
        get_rag_service._instance = create_vietnamese_rag_service()
    return get_rag_service._instance
