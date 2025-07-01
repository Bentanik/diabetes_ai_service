"""
RAG API routes cho knowledge base management và document processing
"""

import os
import tempfile
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Union

from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Query, status
from fastapi.responses import JSONResponse

from features.rag.rag_pipeline import RAGPipeline, RAGPipelineConfig
from features.rag.chunking import ChunkingConfig
from features.rag.embedding import EmbeddingConfig
from features.rag.vector_store import VectorStoreConfig, VectorStore
from features.rag.storage import document_storage
from core.logging_config import get_logger
from .models import (
    KnowledgeBaseCreate,
    KnowledgeBaseResponse,
    KnowledgeBaseList,
    FileUploadResponse,
    FileInfoModel,
)

router = APIRouter(tags=["RAG Knowledge Base"])
logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".html", ".htm", ".csv", ".md"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

# Singleton pipeline
rag_pipeline: RAGPipeline | None = None


def get_rag_pipeline(collection_name: str) -> RAGPipeline:
    """Get RAG pipeline instance với collection name cụ thể."""
    global rag_pipeline
    if not rag_pipeline:
        config = RAGPipelineConfig(
            chunking_config=ChunkingConfig(),
            embedding_config=EmbeddingConfig(),
            vector_store_config=VectorStoreConfig(collection_name=collection_name),
        )
        rag_pipeline = RAGPipeline(config)
    else:
        # Update collection name nếu khác
        if rag_pipeline.vector_store.config.collection_name != collection_name:
            rag_pipeline.vector_store.config.collection_name = collection_name
    return rag_pipeline


def validate_file(file: UploadFile) -> tuple[bool, str]:
    """Validate file type và size"""
    if not file.filename:
        return False, "Filename is required"

    ext = os.path.splitext(file.filename.lower())[1]
    if ext not in SUPPORTED_EXTENSIONS:
        return (
            False,
            f"Unsupported file type. Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    return True, "Valid"


def format_file_info(
    file: UploadFile, size: int, storage_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Format file info cho response"""
    info = {
        "filename": file.filename,
        "file_size": size,
        "file_extension": (
            os.path.splitext(file.filename.lower())[1] if file.filename else ""
        ),
        "content_type": file.content_type or "application/octet-stream",
        "upload_time": datetime.now().isoformat(),
    }

    if storage_info:
        info.update(
            {
                "storage_path": storage_info["storage_path"],
                "storage_time": storage_info["storage_time"],
            }
        )

    return info


@router.post(
    "/knowledge-bases",
    response_model=KnowledgeBaseResponse,
    summary="📚 Tạo knowledge base mới",
    description="Tạo knowledge base mới trong vector store. Mỗi knowledge base là một collection riêng biệt.",
)
async def create_knowledge_base(kb_data: KnowledgeBaseCreate):
    """
    Tạo knowledge base mới trong vector store.

    - **name**: Tên của knowledge base, sẽ được dùng làm collection name
    - **description**: Mô tả về knowledge base (optional)
    - **metadata**: Metadata bổ sung (optional)
    """
    try:
        # Khởi tạo VectorStore với collection name mới
        config = VectorStoreConfig(collection_name=kb_data.name)
        vector_store = VectorStore(config=config)
        success = vector_store.create_collection(force_recreate=True)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Không thể tạo collection trong vector store",
            )

        info = vector_store.get_collection_info()

        # Đảm bảo luôn có created_at
        created_at = info.get("created_at")
        if not created_at:
            created_at = datetime.now().isoformat()

        return KnowledgeBaseResponse(
            name=kb_data.name,
            description=kb_data.description,
            metadata=kb_data.metadata,
            collection_info=info,
            created_at=created_at,
        )

    except Exception as e:
        logger.error(f"Error creating KB: {e}")
        raise HTTPException(500, detail=f"Lỗi khi tạo knowledge base: {str(e)}")


@router.get(
    "/knowledge-bases",
    response_model=KnowledgeBaseList,
    summary="📚 Lấy danh sách knowledge bases",
    description="Lấy danh sách tất cả knowledge bases hiện có.",
)
async def list_knowledge_bases():
    """
    Lấy danh sách tất cả knowledge bases.
    """
    try:
        # Khởi tạo VectorStore không cần embeddings vì chỉ dùng để quản lý collections
        vector_store = VectorStore()
        collections = vector_store.list_collections()

        kbs = []
        for collection in collections:
            # Cập nhật collection name và lấy thông tin
            vector_store.config.collection_name = collection.name
            info = vector_store.get_collection_info()

            # Đảm bảo luôn có created_at
            created_at = info.get("created_at")
            if not created_at:
                created_at = datetime.now().isoformat()

            kbs.append(
                KnowledgeBaseResponse(
                    name=collection.name,
                    description=None,  # Collection info không có description
                    metadata=None,  # Collection info không có metadata
                    collection_info=info,
                    created_at=created_at,
                )
            )

        return KnowledgeBaseList(
            knowledge_bases=kbs,
            total=len(kbs),
        )

    except Exception as e:
        logger.error(f"Error listing KBs: {e}")
        raise HTTPException(
            500, detail=f"Lỗi khi lấy danh sách knowledge bases: {str(e)}"
        )


@router.delete(
    "/knowledge-bases/{name}",
    response_model=Dict[str, Any],
    summary="🗑️ Xóa knowledge base",
    description="Xóa knowledge base và tất cả documents trong đó.",
)
async def delete_knowledge_base(name: str):
    """
    Xóa knowledge base.

    - **name**: Tên của knowledge base cần xóa
    """
    try:
        # Khởi tạo VectorStore với collection name cần xóa
        config = VectorStoreConfig(collection_name=name)
        vector_store = VectorStore(config=config)
        success = vector_store.delete_collection()

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Không thể xóa knowledge base {name}",
            )

        return {
            "success": True,
            "message": f"Đã xóa knowledge base {name}",
            "time": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error deleting KB: {e}")
        raise HTTPException(500, detail=f"Lỗi khi xóa knowledge base: {str(e)}")


@router.post(
    "/knowledge-bases/{name}/documents",
    response_model=FileUploadResponse,
    summary="📄 Upload document vào knowledge base",
    description="Upload và process document vào knowledge base cụ thể.",
)
async def upload_document(
    name: str,
    file: UploadFile = File(...),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    metadata: Dict[str, Any] = Form({}),
):
    """
    Upload và process document vào knowledge base.

    - **name**: Tên của knowledge base
    - **file**: File cần upload
    - **chunk_size**: Kích thước mỗi chunk (default: 1000)
    - **chunk_overlap**: Độ overlap giữa các chunks (default: 200)
    - **metadata**: Metadata bổ sung cho document
    """
    start_time = time.time()
    temp_file = None

    try:
        valid, msg = validate_file(file)
        if not valid:
            raise HTTPException(400, detail=msg)

        content = await file.read()
        size = len(content)
        if size > MAX_FILE_SIZE:
            raise HTTPException(413, detail="File too large.")

        # Store file in MinIO
        storage_info = document_storage.store_document(
            file_data=content,
            filename=file.filename or "unknown",
            knowledge_name=name,
            content_type=file.content_type or "application/octet-stream",
            metadata=metadata,
        )

        # Add storage info to metadata
        metadata.update(
            {
                "storage_path": storage_info["storage_path"],
                "storage_time": storage_info["storage_time"],
            }
        )

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(file.filename or "")[1]
        ) as tmp:
            tmp.write(content)
            temp_file = tmp.name

        # Get pipeline với collection name cụ thể
        pipeline = get_rag_pipeline(name)

        # Update chunk config nếu khác default
        if chunk_size != 1000 or chunk_overlap != 200:
            config = RAGPipelineConfig(
                chunking_config=ChunkingConfig(
                    chunk_size=chunk_size, chunk_overlap=chunk_overlap
                ),
                embedding_config=EmbeddingConfig(),
                vector_store_config=VectorStoreConfig(collection_name=name),
            )
            global rag_pipeline
            rag_pipeline = RAGPipeline(config)
            pipeline = rag_pipeline

        # Add file info vào metadata
        metadata.update(
            {
                "uploaded_filename": file.filename,
                "upload_time": datetime.now().isoformat(),
                "file_size_bytes": size,
                "knowledge_base": name,
            }
        )

        doc_ids = pipeline.process_and_store(temp_file, metadata)
        stats = pipeline.get_stats()
        processing_time = round(time.time() - start_time, 2)

        logger.info(
            f"Uploaded {file.filename} to KB {name}: {len(doc_ids)} vectors in {processing_time}s"
        )

        return FileUploadResponse(
            success=True,
            message=f"Processed {file.filename} -> {len(doc_ids)} vectors",
            file_info=FileInfoModel(**format_file_info(file, size, storage_info)),
            document_ids=doc_ids,
            statistics=stats,
            processing_time=processing_time,
        )

    finally:
        if temp_file and os.path.exists(temp_file):
            os.unlink(temp_file)


@router.get(
    "/knowledge-bases/{name}/stats",
    response_model=Dict[str, Any],
    summary="📊 Knowledge Base Stats",
    description="Lấy thống kê của knowledge base.",
)
async def get_knowledge_base_stats(name: str):
    """
    Lấy thống kê của knowledge base.

    - **name**: Tên của knowledge base
    """
    try:
        pipeline = get_rag_pipeline(name)
        stats = pipeline.get_stats()
        info = pipeline.vector_store.get_collection_info()

        return {
            "success": True,
            "name": name,
            "stats": stats,
            "collection_info": info,
            "time": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error getting KB stats: {e}")
        raise HTTPException(
            500, detail=f"Lỗi khi lấy thống kê knowledge base: {str(e)}"
        )
