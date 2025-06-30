"""
RAG Pipeline - Vietnamese Document Processing

Pipeline để xử lý documents từ raw files đến chunks, 
được tối ưu hóa đặc biệt cho tiếng Việt.

Current Workflow (Phase 1):
1. Document Processing: Parse documents → Extract text + metadata
2. Chunking: Smart chunking với Vietnamese optimization
3. Ready for next phases: Embedding, Vector Storage, Retrieval, Generation

Hỗ trợ:
- Multiple file formats: PDF, DOCX, TXT, HTML, CSV
- Vietnamese text optimization
- Smart chunking strategy selection 
- Batch processing với progress tracking
- Rich metadata tracking

Future phases sẽ có:
- Embedding models integration
- Vector databases (Chroma/FAISS)  
- LLM integration cho Q&A
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from .document_parser import DocumentParser
from .chunking import Chunking, ChunkingConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    RAG Pipeline - Single File Processing

    Hiện tại implement Phase 1: Document Processing + Chunking
    Future phases: Embedding → Vector Storage → Retrieval → Generation

    Features:
    - Single file document parsing (PDF, DOCX, TXT, HTML, CSV)
    - Vietnamese text optimization
    - Smart chunking strategies (Hierarchical, Semantic, Simple)
    - Error handling và robust processing
    - Rich metadata tracking
    - Statistics monitoring
    """

    def __init__(self, chunking_config: Optional[ChunkingConfig] = None):
        """
        Khởi tạo RAG Pipeline.

        Args:
            chunking_config: Cấu hình chunking. Nếu None, dùng config mặc định.
        """
        # Document processing components
        self.parser = DocumentParser()
        self.chunking = Chunking(chunking_config)

        # Statistics tracking
        self.stats = {
            "total_files_processed": 0,
            "total_documents_created": 0,
            "total_chunks_created": 0,
            "processing_errors": 0,
            "last_processing_time": None,
        }

        logger.info("RAG Pipeline initialized successfully")
        logger.info(
            f"Supported formats: {', '.join(sorted(self.parser.supported_formats))}"
        )
        logger.info("Phase 1: Document Processing + Chunking")

    def process_documents(
        self,
        file_path: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Xử lý một file thành chunks.

        Args:
            file_path: Đường dẫn file cần process
            extra_metadata: Metadata bổ sung

        Returns:
            List[Document]: Processed chunks ready for next phase
        """
        return self._process_single_file(file_path, extra_metadata)

    def process_single_document(
        self,
        file_path: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Alias cho process_documents - xử lý một file thành chunks.

        Args:
            file_path: Đường dẫn file cần process
            extra_metadata: Metadata bổ sung

        Returns:
            List[Document]: Processed chunks ready for next phase
        """
        return self.process_documents(file_path, extra_metadata)

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

    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Lấy thống kê quá trình xử lý.

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

        if stats["total_documents_created"] > 0:
            stats["avg_chunks_per_document"] = (
                stats["total_chunks_created"] / stats["total_documents_created"]
            )

        stats["success_rate"] = (
            (stats["total_files_processed"] - stats["processing_errors"])
            / max(stats["total_files_processed"], 1)
        ) * 100

        return stats

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
) -> List[Document]:
    """
    Quick function để process một file với default settings.

    Args:
        file_path: Đường dẫn file
        chunk_size: Kích thước chunk
        chunk_overlap: Overlap giữa chunks

    Returns:
        List[Document]: Chunks được tạo
    """
    config = ChunkingConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    pipeline = RAGPipeline(config)
    return pipeline.process_documents(file_path)


# Backward compatibility aliases
DocumentChunkingPipeline = RAGPipeline
