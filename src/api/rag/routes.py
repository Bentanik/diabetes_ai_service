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
from features.rag.embedding import EmbeddingConfig, Embedding
from features.rag.vector_store import VectorStoreConfig, VectorStore
from features.rag.storage import document_storage
from features.rag.retrieval import (
    HybridRetriever,
    MultiCollectionConfig,
    create_multi_collection_retriever,
    HybridSearchConfig,
    MultiCollectionHybridRetriever,
)
from core.logging_config import get_logger
from .models import (
    KnowledgeBaseCreate,
    KnowledgeBaseResponse,
    KnowledgeBaseList,
    FileUploadResponse,
    FileInfoModel,
    CollectionStats,
    MultiCollectionSearchRequest,
    MultiCollectionSearchResponse,
    SearchResult,
)
from utils.utils import validate_retriever_initialization, check_retriever_type
from features.rag.document_parser import DocumentParser

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
        # Kiểm tra xem collection đã tồn tại chưa
        if collection_name not in rag_pipeline.vector_stores:
            # Tạo vector store mới cho collection
            config = VectorStoreConfig(collection_name=collection_name)
            vector_store = VectorStore(
                embeddings=rag_pipeline.embeddings, config=config
            )
            vector_store.create_collection()
            rag_pipeline.vector_stores[collection_name] = vector_store

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
    description="Xóa knowledge base, tất cả documents và dữ liệu lưu trữ trong MinIO.",
)
async def delete_knowledge_base(name: str):
    """
    Xóa knowledge base.

    - **name**: Tên của knowledge base cần xóa
    """
    try:
        # Xóa collection trong vector store
        config = VectorStoreConfig(collection_name=name)
        vector_store = VectorStore(config=config)
        vector_store_success = vector_store.delete_collection()

        if not vector_store_success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Không thể xóa knowledge base {name} trong vector store",
            )

        # Xóa folder và tất cả documents trong MinIO
        minio_success = document_storage.delete_collection_folder(name)
        if not minio_success:
            logger.warning(f"Không thể xóa documents của collection {name} trong MinIO")

        return {
            "success": True,
            "message": f"Đã xóa knowledge base {name} và tất cả dữ liệu liên quan",
            "time": datetime.now().isoformat(),
            "details": {
                "vector_store_deleted": vector_store_success,
                "minio_documents_deleted": minio_success,
            },
        }

    except Exception as e:
        logger.error(f"Error deleting KB: {e}")
        raise HTTPException(500, detail=f"Lỗi khi xóa knowledge base: {str(e)}")


@router.post(
    "/knowledge-bases/{name}/documents",
    response_model=FileUploadResponse,
    summary="📄 Upload document vào knowledge base",
    description="""Upload và process document vào knowledge base cụ thể.""",
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

        # Store file in MinIO with folder structure
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

        doc_ids = pipeline.process_and_store(
            temp_file, collection_name=name, extra_metadata=metadata
        )
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
        info = pipeline.vector_stores[name].get_collection_info()

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


@router.get(
    "/knowledge-bases/{name}/documents/stats",
    response_model=CollectionStats,
    summary="📊 Thống kê documents trong knowledge base",
    description="Lấy thông tin thống kê về các documents trong một knowledge base.",
)
async def get_documents_stats(name: str):
    """
    Lấy thống kê về documents trong knowledge base.

    - **name**: Tên của knowledge base cần thống kê
    """
    try:
        # Kiểm tra collection có tồn tại không
        vector_store = VectorStore()
        collections = vector_store.list_collections()
        if not any(c.name == name for c in collections):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Không tìm thấy knowledge base {name}",
            )

        # Lấy thống kê từ MinIO
        stats = document_storage.get_collection_stats(name)

        # Format dung lượng để dễ đọc
        total_size_mb = round(stats["total_size_bytes"] / (1024 * 1024), 2)
        logger.info(
            f"Collection {name} stats: {stats['total_documents']} documents, {total_size_mb} MB"
        )

        return CollectionStats(**stats)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting documents stats: {e}")
        raise HTTPException(500, detail=f"Lỗi khi lấy thống kê documents: {str(e)}")


@router.post(
    "/search",
    response_model=MultiCollectionSearchResponse,
    summary="🔍 Tìm kiếm trên nhiều knowledge bases",
    description="Thực hiện tìm kiếm hybrid (BM25 + Vector) trên nhiều knowledge bases.",
)
async def multi_collection_search(request: MultiCollectionSearchRequest):
    """
    Tìm kiếm trên nhiều knowledge bases.

    - **query**: Câu hỏi/query cần tìm kiếm
    - **collection_names**: Danh sách các knowledge bases cần tìm kiếm
    - **top_k**: Số lượng kết quả trả về (default: 5)
    - **score_threshold**: Ngưỡng điểm tối thiểu (default: 0.3)
    """
    try:
        start_time = time.time()

        # Kiểm tra các collections có tồn tại không
        vector_store = VectorStore()
        collections = vector_store.list_collections()
        existing_collections = {c.name for c in collections}

        invalid_collections = set(request.collection_names) - existing_collections
        if invalid_collections:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Không tìm thấy knowledge bases: {', '.join(invalid_collections)}",
            )

        # Khởi tạo embedding model
        try:
            embeddings = Embedding()
            logger.info("Successfully initialized embedding model")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Lỗi khởi tạo embedding model: {str(e)}",
            )

        # Tạo vector stores cho từng collection
        vector_stores = {}
        for name in request.collection_names:
            try:
                config = VectorStoreConfig(collection_name=name)
                store = VectorStore(embeddings=embeddings, config=config)
                store.create_collection()  # Kết nối tới collection đã tồn tại
                vector_stores[name] = store
                logger.info(f"Successfully created vector store for collection: {name}")
            except Exception as e:
                logger.error(
                    f"Failed to create vector store for collection {name}: {e}"
                )
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Lỗi tạo vector store cho collection {name}: {str(e)}",
                )

        # Validate vector stores were created properly
        if not vector_stores:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Không thể tạo được vector stores nào",
            )

        # Tạo retriever cho multi-collection search
        try:
            hybrid_config = HybridSearchConfig(
                bm25_weight=0.3,
                vector_weight=0.7,
                top_k=request.top_k or 5,
                score_threshold=(
                    request.score_threshold
                    if request.score_threshold is not None
                    else 0.3
                ),
            )

            multi_config = MultiCollectionConfig(
                normalize_scores=True,
                top_k=request.top_k or 5,
                score_threshold=(
                    request.score_threshold
                    if request.score_threshold is not None
                    else 0.3
                ),
            )

            logger.info(
                f"Creating MultiCollectionHybridRetriever with {len(vector_stores)} collections"
            )
            retriever = MultiCollectionHybridRetriever(
                vector_stores=vector_stores,
                hybrid_config=hybrid_config,
                multi_collection_config=multi_config,
            )

            # Validate retriever was created properly using utility function
            is_valid, error_msg = validate_retriever_initialization(retriever)
            if not is_valid:
                raise ValueError(f"Retriever validation failed: {error_msg}")

            # Log retriever type information for debugging
            retriever_info = check_retriever_type(retriever)
            logger.info(f"Successfully created retriever: {retriever_info}")

        except Exception as e:
            logger.error(f"Failed to create retriever: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Lỗi tạo retriever: {str(e)}",
            )

        # Thực hiện tìm kiếm
        try:
            logger.info(f"Starting search with query: {request.query[:50]}...")
            results = retriever.get_relevant_documents(request.query)
            logger.info(f"Search completed, found {len(results)} results")
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Lỗi thực hiện tìm kiếm: {str(e)}",
            )

        # Format kết quả
        search_results = []
        for doc in results:
            search_results.append(
                SearchResult(
                    content=doc.page_content,
                    metadata=doc.metadata,
                    score=float(doc.metadata.get("hybrid_score", 0.0)),
                    collection_name=doc.metadata.get("collection_name", "unknown"),
                )
            )

        # Lấy thống kê theo collection
        collection_stats = {}
        for name in request.collection_names:
            try:
                stats = document_storage.get_collection_stats(name)
                collection_stats[name] = stats
            except Exception as e:
                logger.warning(f"Không lấy được thống kê cho {name}: {e}")
                collection_stats[name] = {"error": str(e)}

        processing_time = time.time() - start_time

        return MultiCollectionSearchResponse(
            results=search_results,
            total_results=len(search_results),
            processing_time=processing_time,
            collection_stats=collection_stats,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during multi-collection search: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi thực hiện tìm kiếm: {str(e)}",
        )
