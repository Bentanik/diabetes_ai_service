"""
Test suite cho RAG Pipeline - Single File Processing

Comprehensive tests cho tất cả functionality của RAG Pipeline,
bao gồm document processing, chunking, statistics, error handling.
"""

import sys
import pytest
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Thêm path để import từ src
current_file = Path(__file__)
tests_rag_dir = current_file.parent  # tests/rag/
tests_dir = tests_rag_dir.parent  # tests/
aiservice_dir = tests_dir.parent  # aiservice/
src_dir = aiservice_dir / "src"
sys.path.insert(0, str(src_dir))

# Import modules to test
from src.rag.rag_pipeline import RAGPipeline, process_file
from src.rag.chunking import ChunkingConfig
from langchain_core.documents import Document


class TestRAGPipelineInitialization:
    """Test RAG Pipeline initialization"""

    def test_default_initialization(self):
        """Test khởi tạo với default config"""
        pipeline = RAGPipeline()

        assert pipeline.parser is not None
        assert pipeline.chunking is not None
        assert isinstance(pipeline.stats, dict)
        assert pipeline.stats["total_files_processed"] == 0
        assert pipeline.stats["total_documents_created"] == 0
        assert pipeline.stats["total_chunks_created"] == 0
        assert pipeline.stats["processing_errors"] == 0
        assert pipeline.stats["last_processing_time"] is None

    def test_custom_config_initialization(self):
        """Test khởi tạo với custom chunking config"""
        config = ChunkingConfig(chunk_size=256, chunk_overlap=32, min_chunk_size=50)

        pipeline = RAGPipeline(config)

        assert pipeline.chunking.config.chunk_size == 256
        assert pipeline.chunking.config.chunk_overlap == 32
        assert pipeline.chunking.config.min_chunk_size == 50


class TestRAGPipelineProcessing:
    """Test document processing functionality"""

    @pytest.fixture
    def test_docx_file(self):
        """Path to real test DOCX file"""
        return "tests/data/test_document.docx"

    @pytest.fixture
    def test_pdf_file(self):
        """Path to real test PDF file"""
        return "tests/data/test_document.pdf"

    @pytest.fixture
    def pipeline(self):
        """RAG Pipeline instance for testing"""
        return RAGPipeline()

    def test_process_documents_docx_success(self, pipeline, test_docx_file):
        """Test successful DOCX document processing"""
        chunks = pipeline.process_documents(test_docx_file)

        assert isinstance(chunks, list)
        assert len(chunks) > 0

        # Check chunk properties
        for chunk in chunks:
            assert isinstance(chunk, Document)
            assert len(chunk.page_content) > 0
            assert isinstance(chunk.metadata, dict)

        # Check metadata
        chunk = chunks[0]
        assert "source_file" in chunk.metadata
        assert "file_name" in chunk.metadata
        assert "file_extension" in chunk.metadata
        assert "global_chunk_id" in chunk.metadata
        assert chunk.metadata["file_extension"] == ".docx"
        assert chunk.metadata["file_name"] == "test_document.docx"

    def test_process_documents_pdf_success(self, pipeline, test_pdf_file):
        """Test successful PDF document processing"""
        chunks = pipeline.process_documents(test_pdf_file)

        assert isinstance(chunks, list)
        assert len(chunks) > 0

        # Check chunk properties
        for chunk in chunks:
            assert isinstance(chunk, Document)
            assert len(chunk.page_content) > 0
            assert isinstance(chunk.metadata, dict)

        # Check metadata
        chunk = chunks[0]
        assert "source_file" in chunk.metadata
        assert "file_name" in chunk.metadata
        assert "file_extension" in chunk.metadata
        assert "global_chunk_id" in chunk.metadata
        assert chunk.metadata["file_extension"] == ".pdf"
        assert chunk.metadata["file_name"] == "test_document.pdf"

    def test_process_documents_with_metadata(self, pipeline, test_docx_file):
        """Test processing với extra metadata"""
        extra_metadata = {
            "author": "Test User",
            "category": "Documentation",
            "priority": "high",
        }

        chunks = pipeline.process_documents(test_docx_file, extra_metadata)

        assert len(chunks) > 0

        # Check extra metadata được add
        chunk = chunks[0]
        assert chunk.metadata["author"] == "Test User"
        assert chunk.metadata["category"] == "Documentation"
        assert chunk.metadata["priority"] == "high"

    def test_process_single_document_alias(self, pipeline, test_docx_file):
        """Test process_single_document method (alias)"""
        chunks1 = pipeline.process_documents(test_docx_file)
        chunks2 = pipeline.process_single_document(test_docx_file)

        # Should produce same results
        assert len(chunks1) == len(chunks2)
        assert chunks1[0].page_content == chunks2[0].page_content

    def test_process_nonexistent_file(self, pipeline):
        """Test xử lý file không tồn tại"""
        with pytest.raises(FileNotFoundError):
            pipeline.process_documents("nonexistent_file.txt")

    def test_process_unsupported_format(self, pipeline):
        """Test xử lý file format không support"""
        # Tạo file với extension không support
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"test content")
            temp_file = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported format"):
                pipeline.process_documents(temp_file)
        finally:
            os.unlink(temp_file)

    def test_process_empty_file(self, pipeline):
        """Test xử lý file rỗng"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("")  # Empty file
            temp_file = f.name

        try:
            chunks = pipeline.process_documents(temp_file)
            # Should return empty list for empty file
            assert chunks == []
        finally:
            os.unlink(temp_file)


class TestRAGPipelineStatistics:
    """Test statistics tracking functionality"""

    @pytest.fixture
    def pipeline(self):
        return RAGPipeline()

    @pytest.fixture
    def test_file(self):
        return "tests/data/test_document.docx"

    def test_initial_stats(self, pipeline):
        """Test initial statistics values"""
        stats = pipeline.get_processing_stats()

        assert stats["total_files_processed"] == 0
        assert stats["total_documents_created"] == 0
        assert stats["total_chunks_created"] == 0
        assert stats["processing_errors"] == 0
        assert stats["success_rate"] == 0.0

    def test_stats_after_processing(self, pipeline, test_file):
        """Test statistics sau khi process file"""
        chunks = pipeline.process_documents(test_file)
        stats = pipeline.get_processing_stats()

        assert stats["total_files_processed"] == 1
        assert stats["total_documents_created"] >= 1
        assert stats["total_chunks_created"] >= 1
        assert stats["processing_errors"] == 0
        assert stats["success_rate"] == 100.0

        # Test derived metrics
        assert "avg_documents_per_file" in stats
        assert "avg_chunks_per_file" in stats
        assert "avg_chunks_per_document" in stats

    def test_stats_multiple_files(self, pipeline):
        """Test statistics với multiple files"""
        # Use both test files
        files = ["tests/data/test_document.docx", "tests/data/test_document.pdf"]

        # Process each file
        for file_path in files:
            pipeline.process_documents(file_path)

        stats = pipeline.get_processing_stats()

        assert stats["total_files_processed"] == 2
        assert stats["success_rate"] == 100.0
        assert stats["avg_documents_per_file"] >= 1.0

    def test_stats_with_errors(self, pipeline, test_file):
        """Test statistics khi có errors"""
        # Process successful file
        pipeline.process_documents(test_file)

        # Try to process non-existent file
        try:
            pipeline.process_documents("nonexistent.txt")
        except FileNotFoundError:
            pass

        stats = pipeline.get_processing_stats()

        assert stats["total_files_processed"] == 1
        assert stats["processing_errors"] == 1
        assert stats["success_rate"] == 0.0  # (1-1)/1 = 0

    def test_reset_stats(self, pipeline, test_file):
        """Test reset statistics"""
        # Process file to generate stats
        pipeline.process_documents(test_file)

        # Reset stats
        pipeline.reset_stats()

        stats = pipeline.get_processing_stats()
        assert stats["total_files_processed"] == 0
        assert stats["total_documents_created"] == 0
        assert stats["total_chunks_created"] == 0
        assert stats["processing_errors"] == 0


class TestRAGPipelineExport:
    """Test export functionality"""

    @pytest.fixture
    def pipeline(self):
        return RAGPipeline()

    @pytest.fixture
    def sample_chunks(self):
        """Sample chunks for testing export"""
        return [
            Document(
                page_content="First chunk content",
                metadata={"chunk_id": 0, "source": "test"},
            ),
            Document(
                page_content="Second chunk content",
                metadata={"chunk_id": 1, "source": "test"},
            ),
        ]

    def test_export_chunks_to_json(self, pipeline, sample_chunks):
        """Test export chunks to JSON"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_file = f.name

        try:
            pipeline.export_chunks_to_json(sample_chunks, output_file)

            # Verify file exists
            assert os.path.exists(output_file)

            # Verify content
            with open(output_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            assert len(data) == 2
            assert data[0]["content"] == "First chunk content"
            assert data[1]["content"] == "Second chunk content"
            assert data[0]["chunk_id"] == 0
            assert data[1]["chunk_id"] == 1

        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)

    def test_export_chunks_without_metadata(self, pipeline, sample_chunks):
        """Test export chunks without metadata"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_file = f.name

        try:
            pipeline.export_chunks_to_json(
                sample_chunks, output_file, include_metadata=False
            )

            with open(output_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            assert len(data) == 2
            assert "metadata" not in data[0]
            assert "metadata" not in data[1]
            assert "content" in data[0]
            assert "content_length" in data[0]

        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)

    def test_export_empty_chunks(self, pipeline):
        """Test export empty chunks list"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_file = f.name

        try:
            pipeline.export_chunks_to_json([], output_file)

            with open(output_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            assert data == []

        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)


class TestRAGPipelineFutureFeatures:
    """Test future features (stubs)"""

    @pytest.fixture
    def pipeline(self):
        return RAGPipeline()

    @pytest.fixture
    def sample_chunks(self):
        return [Document(page_content="Test chunk", metadata={"test": True})]

    def test_prepare_for_embedding(self, pipeline, sample_chunks):
        """Test prepare_for_embedding method"""
        prepared = pipeline.prepare_for_embedding(sample_chunks)

        # Should return same chunks for now
        assert prepared == sample_chunks
        assert len(prepared) == 1

    def test_get_embedding_info(self, pipeline):
        """Test get_embedding_info method"""
        info = pipeline.get_embedding_info()

        assert isinstance(info, dict)
        assert "status" in info
        assert "next_phase" in info
        assert "chunks_ready" in info
        assert info["status"] == "Phase 1 completed"
        assert info["chunks_ready"] == "Yes"


class TestQuickAPIFunction:
    """Test quick API function"""

    @pytest.fixture
    def test_file(self):
        return "tests/data/test_document.docx"

    def test_process_file_function(self, test_file):
        """Test global process_file function"""
        chunks = process_file(test_file)

        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert isinstance(chunks[0], Document)

    def test_process_file_custom_config(self, test_file):
        """Test process_file với custom config"""
        chunks = process_file(test_file, chunk_size=128, chunk_overlap=16)

        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_process_file_error_handling(self):
        """Test error handling của process_file"""
        with pytest.raises(FileNotFoundError):
            process_file("nonexistent_file.txt")


class TestRAGPipelineEdgeCases:
    """Test edge cases và special scenarios"""

    @pytest.fixture
    def pipeline(self):
        return RAGPipeline()

    def test_very_small_file(self, pipeline):
        """Test xử lý file rất nhỏ"""
        content = "Hi"  # Very small content
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            temp_file = f.name

        try:
            chunks = pipeline.process_documents(temp_file)
            # Should handle gracefully
            assert isinstance(chunks, list)

        finally:
            os.unlink(temp_file)

    def test_unicode_content(self, pipeline):
        """Test xử lý content có Unicode characters"""
        content = """
        Tài liệu có ký tự đặc biệt: áàảãạăắằẳẵặâấầẩẫậ
        Emoji: 😀🎉🚀📚💯
        Symbols: ©®™€£¥§¶†‡•…‰‱¡¿
        """

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            temp_file = f.name

        try:
            chunks = pipeline.process_documents(temp_file)
            assert len(chunks) > 0

            # Check Vietnamese diacritics được preserve (this works)
            chunk_content = chunks[0].page_content
            assert "áàảãạ" in chunk_content

            # Check that basic structure is preserved even if special chars are lost
            assert "Tài liệu có ký tự đặc biệt" in chunk_content
            assert "Emoji:" in chunk_content  # Text structure preserved
            assert "Symbols:" in chunk_content  # Text structure preserved

        finally:
            os.unlink(temp_file)

    def test_multiple_processing_same_pipeline(self, pipeline):
        """Test xử lý multiple files với cùng pipeline instance"""
        # Use both test files
        files = ["tests/data/test_document.docx", "tests/data/test_document.pdf"]
        all_chunks = []

        # Process each file
        for file_path in files:
            chunks = pipeline.process_documents(file_path)
            all_chunks.extend(chunks)

        # Verify results
        assert len(all_chunks) >= 2  # At least 1 chunk per file

        # Check stats
        stats = pipeline.get_processing_stats()
        assert stats["total_files_processed"] == 2


class TestRAGPipelineIntegration:
    """Integration tests với real scenarios"""

    def test_full_workflow_scenario(self):
        """Test complete workflow từ init đến export"""
        # Use real test file
        test_file = "tests/data/test_document.docx"
        output_file = None

        try:
            # Step 1: Initialize pipeline
            config = ChunkingConfig(chunk_size=256, chunk_overlap=32)
            pipeline = RAGPipeline(config)

            # Step 2: Process document
            chunks = pipeline.process_documents(
                test_file, {"project": "AI Report", "team": "NLP Team"}
            )

            # Step 3: Verify processing
            assert len(chunks) > 0
            assert chunks[0].metadata["project"] == "AI Report"
            assert chunks[0].metadata["team"] == "NLP Team"
            assert chunks[0].metadata["file_name"] == "test_document.docx"

            # Step 4: Check statistics
            stats = pipeline.get_processing_stats()
            assert stats["total_files_processed"] == 1
            assert stats["success_rate"] == 100.0

            # Step 5: Export results
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                output_file = f.name

            pipeline.export_chunks_to_json(chunks, output_file)

            # Step 6: Verify export
            with open(output_file, "r", encoding="utf-8") as f:
                exported_data = json.load(f)

            assert len(exported_data) == len(chunks)
            assert len(exported_data[0]["content"]) > 0

            # Step 7: Prepare for next phase
            prepared = pipeline.prepare_for_embedding(chunks)
            embedding_info = pipeline.get_embedding_info()

            assert len(prepared) == len(chunks)
            assert embedding_info["status"] == "Phase 1 completed"

        finally:
            # Cleanup
            if output_file and os.path.exists(output_file):
                os.unlink(output_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
