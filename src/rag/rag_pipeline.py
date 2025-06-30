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
from .embedding import EmbeddingConfig, MultilinguaE5Embeddings
from .vector_store import VectorStoreConfig, QdrantVectorService

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
        self.embeddings = MultilinguaE5Embeddings(self.config.embedding_config)
        self.vector_store = QdrantVectorService(
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
        stats = self.stats.copy()

        # Add derived metrics
        if stats["total_files_processed"] > 0:
            stats["avg_documents_per_file"] = (
                stats["total_documents_created"] / stats["total_files_processed"]
            )
            stats["avg_chunks_per_file"] = (
                stats["total_chunks_created"] / stats["total_files_processed"]
            )
            stats["avg_vectors_per_file"] = (
                stats["total_vectors_stored"] / stats["total_files_processed"]
            )

        if stats["total_documents_created"] > 0:
            stats["avg_chunks_per_document"] = (
                stats["total_chunks_created"] / stats["total_documents_created"]
            )

        stats["success_rate"] = (
            (stats["total_files_processed"] - stats["processing_errors"])
            / max(stats["total_files_processed"], 1)
        ) * 100

        # Add vector store stats
        vector_store_stats = self.vector_store.get_stats()
        stats.update({"vector_store_stats": vector_store_stats})

        return stats

    def _process_single_file(
        self, file_path: str, extra_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Xử lý một file duy nhất từ parsing đến chunking."""
        start_time = datetime.now()

        try:
            # Validate file
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            file_path_obj = Path(file_path)
            if file_path_obj.suffix.lower() not in self.parser.supported_formats:
                raise ValueError(f"Unsupported format: {file_path_obj.suffix}")

            logger.info(f"Processing file: {file_path_obj.name}")
            # Step 1: Parse document
            documents = self.parser.load_single_document(str(file_path))

            if not documents:
                logger.warning(f"No content extracted from {Path(file_path).name}")
                return []

            logger.info(
                f"Parsed {len(documents)} document(s) from {Path(file_path).name}"
            )

            # Step 2: Add file-level metadata
            file_metadata = {
                "source_file": str(file_path),
                "file_name": Path(file_path).name,
                "file_extension": Path(file_path).suffix.lower(),
                "file_size": Path(file_path).stat().st_size,
                "processing_timestamp": start_time.isoformat(),
            }

            if extra_metadata:
                file_metadata.update(extra_metadata)

            # Add file metadata to all documents
            for doc in documents:
                doc.metadata.update(file_metadata)

            # Step 3: Chunk documents
            chunks = self.chunking.chunk_documents(documents)

            # Step 4: Add chunk-level metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update(
                    {
                        "global_chunk_id": f"{file_path_obj.stem}_chunk_{i}",
                        "source_document_count": len(documents),
                        "total_chunks_in_file": len(chunks),
                    }
                )

            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            self.stats["total_files_processed"] += 1
            self.stats["total_documents_created"] += len(documents)
            self.stats["total_chunks_created"] += len(chunks)
            self.stats["last_processing_time"] = processing_time

            logger.info(
                f"File {Path(file_path).name} → {len(chunks)} chunks ({processing_time:.2f}s)"
            )
            return chunks

        except Exception as e:
            self.stats["processing_errors"] += 1
            logger.error(f"Error processing {Path(file_path).name}: {str(e)}")
            raise

    def export_chunks_to_json(
        self, chunks: List[Document], output_path: str, include_metadata: bool = True
    ) -> None:
        """
        Export chunks ra file JSON để debug hoặc analyze.

        Args:
            chunks: Danh sách chunks cần export
            output_path: Đường dẫn file JSON output
            include_metadata: Có include metadata không
        """
        export_data = []

        for i, chunk in enumerate(chunks):
            chunk_data = {
                "chunk_id": i,
                "content": chunk.page_content,
                "content_length": len(chunk.page_content),
            }

            if include_metadata:
                chunk_data["metadata"] = chunk.metadata

            export_data.append(chunk_data)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Exported {len(chunks)} chunks to {output_path}")

    def reset_stats(self) -> None:
        """Reset tất cả statistics về 0."""
        self.stats = {
            "total_files_processed": 0,
            "total_documents_created": 0,
            "total_chunks_created": 0,
            "total_embeddings_created": 0,
            "total_vectors_stored": 0,
            "processing_errors": 0,
            "last_processing_time": None,
        }
        logger.info("Statistics reset")

    # Future methods - stub implementation
    def prepare_for_embedding(self, chunks: List[Document]) -> List[Document]:
        """
        Prepare chunks for embedding phase (Phase 2).

        Args:
            chunks: Processed chunks

        Returns:
            Chunks ready for embedding
        """
        logger.info(f"Prepared {len(chunks)} chunks for embedding phase")
        return chunks

    def get_embedding_info(self) -> Dict[str, str]:
        """Get information about embedding readiness."""
        return {
            "status": "Phase 1 completed",
            "next_phase": "Embedding implementation",
            "chunks_ready": "Yes",
            "note": "Ready for vector embedding integration",
        }


# Simple API function cho quick usage
def process_file(
    file_path: str, chunk_size: int = 512, chunk_overlap: int = 64
) -> List[str]:
    """
    Quick function để process một file với default settings.

    Args:
        file_path: Đường dẫn file
        chunk_size: Kích thước chunk
        chunk_overlap: Overlap giữa chunks

    Returns:
        List[str]: Document IDs trong vector store
    """
    chunking_config = ChunkingConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    config = RAGPipelineConfig(chunking_config=chunking_config)
    pipeline = RAGPipeline(config)
    return pipeline.process_and_store(file_path)


# Backward compatibility aliases
DocumentChunkingPipeline = RAGPipeline
