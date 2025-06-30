"""RAG API routes cho upload và processing files."""

import os
import tempfile
import time
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse

from rag.rag_pipeline import RAGPipeline, ChunkingConfig
from core.logging_config import get_logger
from .models import FileUploadResponse, ErrorResponse, FileInfoModel

# Setup
router = APIRouter(tags=["RAG Document Processing"])
logger = get_logger(__name__)

# Supported file types
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".html", ".htm", ".csv", ".md"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# Global RAG pipeline instance
rag_pipeline = None


def get_rag_pipeline() -> RAGPipeline:
    """Get or create RAG pipeline instance."""
    global rag_pipeline
    if rag_pipeline is None:
        # Default config - có thể customize sau
        config = ChunkingConfig(chunk_size=1000, chunk_overlap=200, min_chunk_size=50)
        rag_pipeline = RAGPipeline(config)
        logger.info("RAG Pipeline initialized successfully")
    return rag_pipeline


def validate_file(file: UploadFile) -> tuple[bool, str]:
    """Validate uploaded file."""
    # Check file extension
    if not file.filename:
        return False, "Filename is required"

    file_ext = os.path.splitext(file.filename.lower())[1]
    if file_ext not in SUPPORTED_EXTENSIONS:
        return (
            False,
            f"Unsupported file type. Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    # Check file size (rough estimate based on content length)
    if hasattr(file, "size") and file.size is not None and file.size > MAX_FILE_SIZE:
        return False, f"File too large. Max size: {MAX_FILE_SIZE // (1024*1024)}MB"

    return True, "Valid"


def format_file_info(file: UploadFile, file_size: int) -> Dict[str, Any]:
    """Format file information."""
    return {
        "filename": file.filename,
        "file_size": file_size,
        "file_extension": (
            os.path.splitext(file.filename.lower())[1] if file.filename else ""
        ),
        "content_type": file.content_type or "application/octet-stream",
        "upload_time": datetime.now().isoformat(),
    }


def format_chunks_response(chunks) -> list[Dict[str, Any]]:
    """Format chunks for response."""
    formatted_chunks = []
    for chunk in chunks:
        formatted_chunks.append(
            {
                "content": chunk.page_content,
                "content_length": len(chunk.page_content),
                "metadata": chunk.metadata,
            }
        )
    return formatted_chunks


@router.post(
    "/upload",
    response_model=FileUploadResponse,
    summary="📄 Upload và xử lý tài liệu",
    description="""
    Upload file và xử lý bằng RAG Pipeline:
    
    - **Hỗ trợ formats**: PDF, DOCX, TXT, HTML, CSV, MD
    - **Max file size**: 50MB
    - **Output**: Chunks đã được xử lý với metadata
    - **Vietnamese optimized**: Tối ưu cho tiếng Việt
    """,
)
async def upload_file(
    file: UploadFile = File(..., description="File để upload và xử lý"),
    chunk_size: int = Form(1000, description="Kích thước chunk (default: 1000)"),
    chunk_overlap: int = Form(200, description="Overlap giữa chunks (default: 200)"),
    metadata: str = Form(None, description="Extra metadata (JSON string, optional)"),
):
    """
    Upload và xử lý file bằng RAG Pipeline.

    Returns:
        FileUploadResponse: Kết quả processing với chunks và statistics
    """
    start_time = time.time()
    temp_file_path = None

    try:
        # Validate file
        is_valid, error_msg = validate_file(file)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)

        # Read file content
        file_content = await file.read()
        file_size = len(file_content)

        # Check actual file size
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large: {file_size // (1024*1024)}MB. Max: {MAX_FILE_SIZE // (1024*1024)}MB",
            )

        # Create temporary file
        file_ext = os.path.splitext(file.filename.lower())[1] if file.filename else ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name

        # Get RAG pipeline
        pipeline = get_rag_pipeline()

        # Update config if specified
        if chunk_size != 1000 or chunk_overlap != 200:
            new_config = ChunkingConfig(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap, min_chunk_size=50
            )
            pipeline = RAGPipeline(new_config)

        # Parse extra metadata
        extra_metadata = {}
        if metadata:
            try:
                import json

                extra_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                logger.warning(f"Invalid metadata JSON: {metadata}")

        # Add upload info to metadata
        extra_metadata.update(
            {
                "uploaded_filename": file.filename,
                "upload_time": datetime.now().isoformat(),
                "file_size_bytes": file_size,
            }
        )

        # Process document
        logger.info(f"Processing file: {file.filename} ({file_size} bytes)")
        chunks = pipeline.process_documents(temp_file_path, extra_metadata)

        # Get statistics
        stats = pipeline.get_processing_stats()
        processing_time = time.time() - start_time

        # Format file info
        file_info = format_file_info(file, file_size)

        # Format chunks
        formatted_chunks = format_chunks_response(chunks)

        logger.info(
            f"Successfully processed {file.filename}: {len(chunks)} chunks in {processing_time:.2f}s"
        )

        return FileUploadResponse(
            success=True,
            message=f"Successfully processed {file.filename}. Created {len(chunks)} chunks.",
            file_info=file_info,
            chunks=formatted_chunks,
            statistics=stats,
            processing_time=round(processing_time, 2),
        )

    except HTTPException:
        raise
    except FileNotFoundError as e:
        logger.error(f"File not found error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"File processing error: {str(e)}")
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error processing {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        # Cleanup temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(
                    f"Failed to cleanup temp file {temp_file_path}: {str(e)}"
                )


@router.get(
    "/stats",
    summary="📊 Xem thống kê RAG Pipeline",
    description="Lấy thống kê chi tiết về việc xử lý documents",
)
async def get_pipeline_stats():
    """Lấy thống kê từ RAG Pipeline."""
    try:
        pipeline = get_rag_pipeline()
        stats = pipeline.get_processing_stats()
        return {
            "success": True,
            "statistics": stats,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error getting statistics: {str(e)}"
        )


@router.post(
    "/reset-stats",
    summary="🔄 Reset thống kê",
    description="Reset tất cả thống kê của RAG Pipeline",
)
async def reset_pipeline_stats():
    """Reset thống kê RAG Pipeline."""
    try:
        pipeline = get_rag_pipeline()
        pipeline.reset_stats()
        return {
            "success": True,
            "message": "Statistics reset successfully",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error resetting stats: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error resetting statistics: {str(e)}"
        )


@router.get(
    "/health", summary="🏥 Health check", description="Kiểm tra trạng thái RAG Pipeline"
)
async def health_check():
    """Health check cho RAG Pipeline."""
    try:
        pipeline = get_rag_pipeline()
        return {
            "success": True,
            "status": "healthy",
            "message": "RAG Pipeline is running",
            "supported_formats": list(SUPPORTED_EXTENSIONS),
            "max_file_size_mb": MAX_FILE_SIZE // (1024 * 1024),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Service unhealthy: {str(e)}")
