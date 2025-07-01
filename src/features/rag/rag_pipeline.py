"""
RAG Pipeline - Vietnamese Document Processing

Pipeline để xử lý documents từ raw files đến chunks, 
được tối ưu hóa đặc biệt cho tiếng Việt.

Current Workflow:
1. Document Processing: Parse documents → Extract text + metadata
2. Chunking: Smart chunking với Vietnamese optimization
3. Embedding: Multilingual E5 embeddings
4. Vector Storage: Qdrant vector store
5. Ready for next phases: Retrieval, Generation

Hỗ trợ:
- Multiple file formats: PDF, DOCX, TXT, HTML, CSV
- Vietnamese text optimization
- Smart chunking strategy selection 
- Batch processing với progress tracking
- Rich metadata tracking
- Multilingual embeddings
- Vector storage và retrieval

Future phases sẽ có:
- Embedding models integration
- Vector databases (Chroma/FAISS)  
- LLM integration cho Q&A
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
import json
from dataclasses import dataclass

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from .document_parser import DocumentParser
from .chunking import Chunking, ChunkingConfig
from .embedding import EmbeddingConfig, Embedding
from .vector_store import VectorStoreConfig, VectorStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RAGPipelineConfig:
    """
    Cấu hình cho toàn bộ RAG pipeline.

    Attributes:
        chunking_config: Cấu hình cho chunking phase
        embedding_config: Cấu hình cho embedding phase
        vector_store_config: Cấu hình cho vector store phase
    """

    chunking_config: Optional[ChunkingConfig] = None
    embedding_config: Optional[EmbeddingConfig] = None
    vector_store_config: Optional[VectorStoreConfig] = None

    def __post_init__(self):
        """Set default configs nếu chưa được cung cấp."""
        if self.chunking_config is None:
            self.chunking_config = ChunkingConfig()
        if self.embedding_config is None:
            self.embedding_config = EmbeddingConfig()
        if self.vector_store_config is None:
            self.vector_store_config = VectorStoreConfig()


class RAGPipeline:
    """
    RAG Pipeline - Single File Processing

    Features:
    - Single file document parsing (PDF, DOCX, TXT, HTML, CSV)
    - Vietnamese text optimization
    - Smart chunking strategies (Hierarchical, Semantic, Simple)
    - Multilingual embeddings (E5 model)
    - Vector storage (Qdrant)
    - Error handling và robust processing
    - Rich metadata tracking
    - Statistics monitoring
    """

    def __init__(self, config: Optional[RAGPipelineConfig] = None):
        """
        Khởi tạo RAG Pipeline.

        Args:
            config: Cấu hình cho pipeline. Nếu None, dùng config mặc định.
        """
        # Load config
        self.config = config or RAGPipelineConfig()

        # Document processing components
        self.parser = DocumentParser()
        self.chunking = Chunking(self.config.chunking_config)

        # Embedding và vector store components
        self.embeddings = Embedding(self.config.embedding_config)
        self.vector_store = VectorStore(
            embeddings=self.embeddings, config=self.config.vector_store_config
        )

        # Statistics tracking
        self.stats = {
            "total_files_processed": 0,
            "total_documents_created": 0,
            "total_chunks_created": 0,
            "total_embeddings_created": 0,
            "total_vectors_stored": 0,
            "processing_errors": 0,
            "last_processing_time": None,
        }

        # Initialize vector store collection
        self.vector_store.create_collection()

        logger.info("RAG Pipeline initialized successfully")
        logger.info(
            f"Supported formats: {', '.join(sorted(self.parser.supported_formats))}"
        )
        logger.info(
            "All phases ready: Document Processing + Chunking + Embedding + Vector Storage"
        )

    def process_and_store(
        self,
        file_path: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Process một file và lưu vào vector store.

        Args:
            file_path: Đường dẫn file cần process
            extra_metadata: Metadata bổ sung

        Returns:
            List[str]: Document IDs trong vector store
        """
        # Process file thành chunks
        chunks = self._process_single_file(file_path, extra_metadata)
        if not chunks:
            return []

        # Add embedding metadata
        embedding_metadata = self.embeddings.get_metadata()
        for chunk in chunks:
            chunk.metadata.update(
                {
                    "embedding_model": embedding_metadata["model_name"],
                    "embedding_dimension": embedding_metadata["dimension"],
                }
            )

        # Store trong vector store
        try:
            document_ids = self.vector_store.add_documents(chunks)
            self.stats["total_embeddings_created"] += len(chunks)
            self.stats["total_vectors_stored"] += len(document_ids)
            logger.info(
                f"Stored {len(document_ids)} vectors in {self.vector_store.config.collection_name}"
            )
            return document_ids
        except Exception as e:
            logger.error(f"Error storing vectors: {str(e)}")
            raise

    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
        return_scores: bool = False,
    ) -> Union[List[Document], List[Tuple[Document, float]]]:
        """
        Tìm kiếm similarity trong vector store.

        Args:
            query: Query text
            k: Số lượng kết quả tối đa
            score_threshold: Ngưỡng similarity score
            filter_conditions: Điều kiện lọc metadata
            return_scores: Có trả về scores không

        Returns:
            Nếu return_scores=False: List[Document]
            Nếu return_scores=True: List[Tuple[Document, float]]
        """
        try:
            if return_scores:
                return self.vector_store.similarity_search_with_score(
                    query=query,
                    k=k,
                    score_threshold=score_threshold,
                    filter_conditions=filter_conditions,
                )
            else:
                return self.vector_store.similarity_search(
                    query=query,
                    k=k,
                    score_threshold=score_threshold,
                    filter_conditions=filter_conditions,
                )
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """
        Lấy thống kê của toàn bộ pipeline.

        Returns:
            Dict với thông tin thống kê đầy đủ
        """
        stats: Dict[str, Any] = {}

        # Add pipeline stats
        pipeline_stats = self.stats.copy()

        # Add derived metrics
        if pipeline_stats["total_files_processed"] > 0:
            pipeline_stats["avg_documents_per_file"] = (
                pipeline_stats["total_documents_created"]
                / pipeline_stats["total_files_processed"]
            )
            pipeline_stats["avg_chunks_per_file"] = (
                pipeline_stats["total_chunks_created"]
                / pipeline_stats["total_files_processed"]
            )
            pipeline_stats["avg_vectors_per_file"] = (
                pipeline_stats["total_vectors_stored"]
                / pipeline_stats["total_files_processed"]
            )

        if pipeline_stats["total_documents_created"] > 0:
            pipeline_stats["avg_chunks_per_document"] = (
                pipeline_stats["total_chunks_created"]
                / pipeline_stats["total_documents_created"]
            )

        pipeline_stats["success_rate"] = (
            (
                pipeline_stats["total_files_processed"]
                - pipeline_stats["processing_errors"]
            )
            / max(pipeline_stats["total_files_processed"], 1)
        ) * 100

        # Add component stats
        for k, v in pipeline_stats.items():
            if isinstance(v, (int, float)):
                stats[str(k)] = v
            else:
                stats[str(k)] = str(v)

        for k, v in self.embeddings.get_stats().items():
            if isinstance(v, (int, float)):
                stats[f"embedding_{str(k)}"] = v
            else:
                stats[f"embedding_{str(k)}"] = str(v)

        for k, v in self.vector_store.get_stats().items():
            if isinstance(v, (int, float)):
                stats[f"vector_store_{str(k)}"] = v
            else:
                stats[f"vector_store_{str(k)}"] = str(v)

        return stats

    def _process_single_file(
        self, file_path: str, extra_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Process một file thành chunks.

        Args:
            file_path: Đường dẫn file cần process
            extra_metadata: Metadata bổ sung

        Returns:
            List[Document]: Chunks đã được process
        """
        start_time = datetime.now()

        try:
            # Parse document
            documents = self.parser.load_single_document(file_path)
            if not documents:
                logger.warning(f"No content extracted from {file_path}")
                return []

            self.stats["total_documents_created"] += len(documents)

            # Add metadata
            if extra_metadata:
                for doc in documents:
                    doc.metadata.update(extra_metadata)

            # Chunk documents
            chunks = []
            for doc in documents:
                doc_chunks = self.chunking.chunk_documents([doc])
                chunks.extend(doc_chunks)

            self.stats["total_chunks_created"] += len(chunks)
            self.stats["total_files_processed"] += 1
            self.stats["last_processing_time"] = (
                datetime.now() - start_time
            ).total_seconds()

            logger.info(
                f"Processed {file_path}: {len(documents)} documents → {len(chunks)} chunks"
            )
            return chunks

        except Exception as e:
            self.stats["processing_errors"] += 1
            logger.error(f"Error processing {file_path}: {str(e)}")
            raise

    def reset_stats(self) -> None:
        """Reset statistics về 0."""
        self.stats = {
            "total_files_processed": 0,
            "total_documents_created": 0,
            "total_chunks_created": 0,
            "total_embeddings_created": 0,
            "total_vectors_stored": 0,
            "processing_errors": 0,
            "last_processing_time": None,
        }
        self.embeddings.reset_stats()
        self.vector_store.reset_stats()
        logger.info("Pipeline statistics reset")

    def prepare_for_embedding(self, chunks: List[Document]) -> List[Document]:
        """
        Chuẩn bị chunks cho embedding phase.

        Args:
            chunks: List chunks cần chuẩn bị

        Returns:
            List[Document]: Chunks đã được chuẩn bị
        """
        return self.embeddings.embed_documents_from_chunks(chunks)

    def get_embedding_info(self) -> Dict[str, str]:
        """
        Lấy thông tin về embedding model.

        Returns:
            Dict với thông tin về model
        """
        return self.embeddings.get_metadata()


def process_file(
    file_path: str, chunk_size: int = 512, chunk_overlap: int = 64
) -> List[str]:
    """
    Hàm tiện ích để process một file với config mặc định.

    Args:
        file_path: Đường dẫn file cần process
        chunk_size: Kích thước chunk
        chunk_overlap: Độ overlap giữa các chunks

    Returns:
        List[str]: Document IDs trong vector store
    """
    config = RAGPipelineConfig(
        chunking_config=ChunkingConfig(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    )
    pipeline = RAGPipeline(config)
    return pipeline.process_and_store(file_path)


# Backward compatibility aliases
DocumentChunkingPipeline = RAGPipeline
