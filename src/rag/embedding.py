"""
Enhanced Embedding Service

Kết hợp RagFlow patterns với LangChain và intfloat/multilingual-e5-base
cho tiếng Việt tối ưu.

Features:
- Multilingual E5 model với tiếng Việt support
- Batch processing tối ưu 
- Token counting & monitoring
- Error handling robust
- LangChain integration ready
- Async support cho performance
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
import time

try:
    from sentence_transformers import SentenceTransformer
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
except ImportError as e:
    logging.error(f"Missing required dependencies: {e}")
    raise

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Cấu hình cho embedding service"""

    model_name: str = "intfloat/multilingual-e5-base"
    batch_size: int = 16  # RagFlow default
    max_tokens: int = 512  # E5 optimal length
    device: str = "auto"  # auto-detect GPU/CPU
    normalize_embeddings: bool = True
    query_instruction: str = "query: "  # E5 format
    passage_instruction: str = "passage: "  # E5 format


class MultilinguaE5Embeddings(Embeddings):
    """
    LangChain-compatible embedding class using multilingual-e5-base

    Optimized for Vietnamese text với RagFlow-inspired patterns.
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self.model = None
        self.device = None
        self._stats = {
            "total_texts_embedded": 0,
            "total_tokens_processed": 0,
            "total_embedding_time": 0.0,
            "batch_count": 0,
        }

        self._initialize_model()
        logger.info(f"E5 Embeddings initialized on {self.device}")

    def _initialize_model(self):
        """Initialize embedding model với device optimization"""
        try:
            # Auto-detect device like RagFlow
            if self.config.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = self.config.device

            logger.info(f"Loading {self.config.model_name} on {self.device}...")

            self.model = SentenceTransformer(self.config.model_name, device=self.device)

            # Configure model settings
            self.model.max_seq_length = self.config.max_tokens

            logger.info(
                f"Model loaded successfully. Max length: {self.config.max_tokens}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise

    def _prepare_texts_for_embedding(
        self, texts: List[str], is_query: bool = False
    ) -> List[str]:
        """
        Prepare texts with E5 instructions và length truncation

        E5 format:
        - Query: "query: your question"
        - Passage: "passage: document content"
        """
        instruction = (
            self.config.query_instruction
            if is_query
            else self.config.passage_instruction
        )

        prepared_texts = []
        for text in texts:
            # Truncate text if too long (simple approach)
            if len(text) > self.config.max_tokens * 4:  # rough estimation
                text = text[: self.config.max_tokens * 4]

            # Add E5 instruction
            prepared_text = f"{instruction}{text}"
            prepared_texts.append(prepared_text)

        return prepared_texts

    def _count_tokens(self, texts: List[str]) -> int:
        """Estimate token count - RagFlow pattern"""
        return sum(len(text.split()) for text in texts)  # Simple approximation

    def _encode_batch(
        self, texts: List[str], is_query: bool = False
    ) -> Tuple[np.ndarray, int]:
        """
        Encode a batch of texts với monitoring

        Returns:
            Tuple of (embeddings, token_count)
        """
        start_time = time.time()

        # Prepare texts with instructions
        prepared_texts = self._prepare_texts_for_embedding(texts, is_query)

        # Count tokens for monitoring
        token_count = self._count_tokens(prepared_texts)

        try:
            if self.model is None:
                raise ValueError("Embedding model is not loaded or initialized.")
            # Generate embeddings
            embeddings = self.model.encode(
                prepared_texts,
                convert_to_numpy=True,
                normalize_embeddings=self.config.normalize_embeddings,
                show_progress_bar=False,  # Quiet for batch processing
            )

            # Update stats
            embed_time = time.time() - start_time
            self._stats["total_texts_embedded"] += len(texts)
            self._stats["total_tokens_processed"] += token_count
            self._stats["total_embedding_time"] += embed_time
            self._stats["batch_count"] += 1

            logger.debug(f"Embedded {len(texts)} texts in {embed_time:.2f}s")

            return embeddings, token_count

        except Exception as e:
            logger.error(f"Error encoding batch: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        LangChain interface: Embed documents (passages)

        Uses RagFlow-style batch processing.
        """
        if not texts:
            return []

        all_embeddings = []
        total_token_count = 0

        # Process in batches like RagFlow
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i : i + self.config.batch_size]

            embeddings, token_count = self._encode_batch(batch, is_query=False)
            all_embeddings.extend(embeddings.tolist())
            total_token_count += token_count

        logger.info(
            f"Embedded {len(texts)} documents. Total tokens: {total_token_count}"
        )
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        LangChain interface: Embed single query
        """
        embeddings, token_count = self._encode_batch([text], is_query=True)

        logger.debug(f"Embedded query. Tokens: {token_count}")
        return embeddings[0].tolist()

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async version for documents - RagFlow task_executor pattern"""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(executor, self.embed_documents, texts)

    async def aembed_query(self, text: str) -> List[float]:
        """Async version for query"""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(executor, self.embed_query, text)

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension"""
        dim = 768  # E5-base default
        if self.model is not None:
            model_dim = self.model.get_sentence_embedding_dimension()
            if model_dim is not None:
                dim = model_dim
        return dim

    def get_stats(self) -> Dict[str, Any]:
        """Get embedding statistics - RagFlow monitoring pattern"""
        stats = self._stats.copy()
        stats["avg_time_per_batch"] = stats["total_embedding_time"] / max(
            stats["batch_count"], 1
        )
        stats["avg_tokens_per_text"] = stats["total_tokens_processed"] / max(
            stats["total_texts_embedded"], 1
        )
        return stats

    def reset_stats(self):
        """Reset statistics"""
        for key in self._stats:
            self._stats[key] = 0


class EmbeddingService:
    """
    High-level embedding service

    Integrates multilingual E5 với Document processing pipeline.
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self.embeddings = MultilinguaE5Embeddings(self.config)

        logger.info("EmbeddingService initialized successfully")

    def embed_documents_from_chunks(self, documents: List[Document]) -> List[Document]:
        """
        Embed chunks và add embeddings to metadata

        RagFlow pattern: add vector field "q_{dim}_vec"
        """
        if not documents:
            return documents

        # Extract text content
        texts = [doc.page_content for doc in documents]

        # Generate embeddings
        embeddings = self.embeddings.embed_documents(texts)
        vector_dim = len(embeddings[0]) if embeddings else 768

        # Add embeddings to document metadata
        for doc, embedding in zip(documents, embeddings):
            # RagFlow pattern: store as "q_{dim}_vec"
            doc.metadata[f"q_{vector_dim}_vec"] = embedding
            doc.metadata["embedding_model"] = self.config.model_name
            doc.metadata["embedding_dim"] = vector_dim

        logger.info(
            f"Added embeddings to {len(documents)} documents (dim={vector_dim})"
        )
        return documents

    async def aembed_documents_from_chunks(
        self, documents: List[Document]
    ) -> List[Document]:
        """Async version"""
        if not documents:
            return documents

        texts = [doc.page_content for doc in documents]
        embeddings = await self.embeddings.aembed_documents(texts)
        vector_dim = len(embeddings[0]) if embeddings else 768

        for doc, embedding in zip(documents, embeddings):
            doc.metadata[f"q_{vector_dim}_vec"] = embedding
            doc.metadata["embedding_model"] = self.config.model_name
            doc.metadata["embedding_dim"] = vector_dim

        return documents

    def embed_query(self, query: str) -> List[float]:
        """Embed single query"""
        return self.embeddings.embed_query(query)

    async def aembed_query(self, query: str) -> List[float]:
        """Async embed query"""
        return await self.embeddings.aembed_query(query)

    def get_langchain_embeddings(self) -> Embeddings:
        """Get LangChain-compatible embeddings object"""
        return self.embeddings

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension"""
        return self.embeddings.get_embedding_dimension()

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        return self.embeddings.get_stats()


# Factory functions for easy usage
def create_e5_embeddings(
    config: Optional[EmbeddingConfig] = None,
) -> MultilinguaE5Embeddings:
    """Create E5 embeddings instance"""
    return MultilinguaE5Embeddings(config)


def create_embedding_service(
    config: Optional[EmbeddingConfig] = None,
) -> EmbeddingService:
    """Create embedding service instance"""
    return EmbeddingService(config)


def create_vietnamese_optimized_embeddings() -> EmbeddingService:
    """Create embeddings optimized for Vietnamese"""
    config = EmbeddingConfig(
        model_name="intfloat/multilingual-e5-base",
        batch_size=16,
        max_tokens=512,
        normalize_embeddings=True,
        query_instruction="query: ",
        passage_instruction="passage: ",
    )
    return EmbeddingService(config)
