"""
RAG API routes cho upload và processing files (Tối ưu)
"""

import os
import tempfile
import time
import uuid
import json
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, File, UploadFile, HTTPException, Form, status
from fastapi.responses import JSONResponse

from rag.rag_pipeline import RAGPipeline, RAGPipelineConfig
from rag.chunking import ChunkingConfig
from rag.embedding import EmbeddingConfig, MultilinguaE5Embeddings
from rag.vector_store import QdrantVectorService, VectorStoreConfig
from core.logging_config import get_logger
from .models import (
    FileUploadResponse,
    FileInfoModel,
    KnowledgeBaseCreate,
    KnowledgeBaseResponse,
)

router = APIRouter(tags=["RAG Document Processing"])
logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".html", ".htm", ".csv", ".md"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

# Singleton pipeline
rag_pipeline: RAGPipeline | None = None


def get_rag_pipeline() -> RAGPipeline:
    global rag_pipeline
    if rag_pipeline is None:
        chunking_config = ChunkingConfig(
            chunk_size=1000, chunk_overlap=200, min_chunk_size=50
        )
        embedding_config = EmbeddingConfig()
        vector_store_config = VectorStoreConfig(collection_name="vietnamese_documents")

        config = RAGPipelineConfig(
            chunking_config=chunking_config,
            embedding_config=embedding_config,
            vector_store_config=vector_store_config,
        )

        rag_pipeline = RAGPipeline(config)
        logger.info("RAG Pipeline initialized")
    return rag_pipeline


def validate_file(file: UploadFile) -> tuple[bool, str]:
    if not file.filename:
        return False, "Filename is required"

    ext = os.path.splitext(file.filename.lower())[1]
    if ext not in SUPPORTED_EXTENSIONS:
        return (
            False,
            f"Unsupported file type. Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    return True, "Valid"


def format_file_info(file: UploadFile, size: int) -> Dict[str, Any]:
    return {
        "filename": file.filename,
        "file_size": size,
        "file_extension": (
            os.path.splitext(file.filename.lower())[1] if file.filename else ""
        ),
        "content_type": file.content_type or "application/octet-stream",
        "upload_time": datetime.now().isoformat(),
    }


@router.post(
    "/knowledge-base",
    response_model=KnowledgeBaseResponse,
    summary="📚 Tạo knowledge base",
)
async def create_knowledge_base(kb_data: KnowledgeBaseCreate):
    """
    Tạo knowledge base mới trong vector store.

    Chỉ cần cung cấp tên, hệ thống sẽ tự tạo collection trong vector store.
    """
    try:
        pipeline = get_rag_pipeline()
        vector_store = pipeline.vector_store

        vector_store.config.collection_name = kb_data.name
        success = vector_store.create_collection()

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Không thể tạo collection trong vector store",
            )

        info = vector_store.get_collection_info()

        return KnowledgeBaseResponse(
            name=kb_data.name,
            collection_name=kb_data.name,
            collection_info=info,
        )

    except Exception as e:
        logger.error(f"Error creating KB: {e}")
        raise HTTPException(500, detail=f"Lỗi khi tạo knowledge base: {str(e)}")


@router.post(
    "/upload",
    response_model=FileUploadResponse,
    summary="📄 Upload & Process Document",
)
async def upload_file(
    file: UploadFile = File(...),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    metadata: str = Form(None),
):
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

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(file.filename or "")[1]
        ) as tmp:
            tmp.write(content)
            temp_file = tmp.name

        pipeline = get_rag_pipeline()

        # Nếu chunk config khác thì tạo pipeline mới và update singleton
        if chunk_size != 1000 or chunk_overlap != 200:
            config = RAGPipelineConfig(
                chunking_config=ChunkingConfig(
                    chunk_size=chunk_size, chunk_overlap=chunk_overlap
                ),
                embedding_config=EmbeddingConfig(),
                vector_store_config=VectorStoreConfig(
                    collection_name="vietnamese_documents"
                ),
            )
            global rag_pipeline
            rag_pipeline = RAGPipeline(config)
            pipeline = rag_pipeline

        extra_metadata = {}
        if metadata:
            try:
                extra_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                logger.warning(f"Invalid metadata: {metadata}")

        extra_metadata.update(
            {
                "uploaded_filename": file.filename,
                "upload_time": datetime.now().isoformat(),
                "file_size_bytes": size,
            }
        )

        doc_ids = pipeline.process_and_store(temp_file, extra_metadata)
        stats = pipeline.get_stats()
        processing_time = round(time.time() - start_time, 2)

        logger.info(
            f"Uploaded {file.filename}: {len(doc_ids)} vectors in {processing_time}s"
        )

        return FileUploadResponse(
            success=True,
            message=f"Processed {file.filename} -> {len(doc_ids)} vectors",
            file_info=FileInfoModel(**format_file_info(file, size)),
            document_ids=doc_ids,
            statistics=stats,
            processing_time=processing_time,
        )

    finally:
        if temp_file and os.path.exists(temp_file):
            os.unlink(temp_file)


@router.get("/stats", summary="📊 RAG Stats")
async def stats():
    pipeline = get_rag_pipeline()
    return {
        "success": True,
        "stats": pipeline.get_stats(),
        "time": datetime.now().isoformat(),
    }


@router.post("/reset-stats", summary="🔄 Reset Stats")
async def reset_stats():
    pipeline = get_rag_pipeline()
    pipeline.reset_stats()
    return {
        "success": True,
        "message": "Stats reset",
        "time": datetime.now().isoformat(),
    }


@router.get("/health", summary="🏥 Health Check")
async def health():
    try:
        pipeline = get_rag_pipeline()
        info = pipeline.vector_store.get_stats()
        return {
            "status": "healthy",
            "vector_store": {
                "collection": pipeline.vector_store.config.collection_name,
                "info": info,
            },
            "time": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}
