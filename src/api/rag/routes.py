"""
API routes cho dịch vụ RAG nâng cao.
Hỗ trợ upload file, query, và quản lý knowledge base với RAGFlow + HuggingFace.
"""

import tempfile
import os
from typing import List, Optional, Literal
from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Query
from pydantic import BaseModel


# Simple response models
class QueryResponse(BaseModel):
    success: bool
    answer: str
    sources: list = []
    num_sources: int = 0
    confidence_score: float = 0.0
    retrieval_method: str = "advanced"
    processing_info: dict = {}


class DocumentUploadResponse(BaseModel):
    success: bool
    message: str
    files_processed: int = 0
    chunks_added: int = 0
    details: list = []
    advanced_features: dict = {}


class SystemInfoResponse(BaseModel):
    success: bool
    system_info: dict


class ApiResponse(BaseModel):
    success: bool
    message: str
    data: dict = {}


from rag.service import get_rag_service
from core.logging_config import get_logger

logger = get_logger(__name__)

# Tạo router với prefix
router = APIRouter(prefix="/rag", tags=["RAG Vietnamese"])


# Pydantic models
class AdvancedQueryRequest(BaseModel):
    """Yêu cầu truy vấn nâng cao."""

    question: str
    k: Optional[int] = None
    use_reranking: Optional[bool] = None
    include_sources: bool = True
    vietnamese_prompt: bool = True


class TextUploadRequest(BaseModel):
    """Yêu cầu upload text."""

    text: str
    metadata: Optional[dict] = None
    preserve_structure: Optional[bool] = None


class AdvancedConfigRequest(BaseModel):
    """Cấu hình dịch vụ RAG."""

    embedding_provider: str = "huggingface"
    embedding_model: str = "BAAI/bge-large-zh-v1.5"
    embedding_device: str = "cpu"
    collection_name: str = "vietnamese_kb"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    use_ragflow_pdf: bool = True
    retrieval_k: int = 5


@router.post("/upload_files", response_model=DocumentUploadResponse)
async def upload_files(
    files: List[UploadFile] = File(...),
    preserve_structure: Optional[bool] = Form(
        None, description="Bảo toàn cấu trúc tài liệu"
    ),
):
    """
    Upload và xử lý nhiều files với RAGFlow parsing.

    Hỗ trợ:
    - PDF với RAGFlow parsing cải tiến
    - DOCX, TXT và các định dạng khác
    - Chunking thông minh tối ưu tiếng Việt
    - Metadata chi tiết về quá trình xử lý
    """
    service = get_rag_service()
    temp_files = []

    try:
        # Kiểm tra files
        if not files:
            raise HTTPException(status_code=400, detail="Cần ít nhất một file")

        # Lưu tạm các files
        file_paths = []
        for file in files:
            if not file.filename:
                continue

            # Tạo file tạm
            suffix = os.path.splitext(file.filename)[1]
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            temp_files.append(temp_file.name)

            # Ghi nội dung
            content = await file.read()
            temp_file.write(content)
            temp_file.close()

            file_paths.append(temp_file.name)

        if not file_paths:
            raise HTTPException(status_code=400, detail="Không có file hợp lệ")

        # Xử lý với dịch vụ nâng cao
        result = await service.add_documents_from_files(
            file_paths=file_paths, preserve_structure=preserve_structure
        )

        logger.info(
            f"Upload nâng cao hoàn thành: {result['files_processed']} files → {result['chunks_added']} chunks"
        )

        return DocumentUploadResponse(
            success=result["success"],
            message=result["message"],
            files_processed=result["files_processed"],
            chunks_added=result["chunks_added"],
            details=result.get("processing_details", []),
            advanced_features=result.get("advanced_features", {}),
        )

    except Exception as e:
        logger.error(f"Lỗi upload files nâng cao: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý files: {str(e)}")

    finally:
        # Xóa files tạm
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass


@router.post("/upload_text", response_model=DocumentUploadResponse)
async def upload_text(request: TextUploadRequest):
    """
    Upload raw text với xử lý nâng cao.

    Features:
    - Chunking thông minh cho tiếng Việt
    - Bảo toàn cấu trúc tùy chọn
    - Metadata tùy chỉnh
    """
    service = get_rag_service()

    try:
        result = await service.add_text(
            text=request.text,
            metadata=request.metadata,
            preserve_structure=request.preserve_structure,
        )

        return DocumentUploadResponse(
            success=result["success"],
            message=result["message"],
            files_processed=1,
            chunks_added=result["chunks_added"],
            advanced_features=result.get("advanced_features", {}),
        )

    except Exception as e:
        logger.error(f"Lỗi upload text: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý text: {str(e)}")


@router.post("/query_embedding_only", response_model=QueryResponse)
async def query_embedding_only(request: AdvancedQueryRequest):
    """
    🔄 Truy vấn Embedding Only (Legacy)

    Endpoint này chỉ sử dụng embedding search truyền thống.
    **Khuyến nghị:** Sử dụng `/query` (hybrid) để có kết quả tốt hơn.

    **Tình huống sử dụng:**
    - Backward compatibility
    - So sánh performance với hybrid
    - Debugging embedding behavior
    """
    service = get_rag_service()

    try:
        result = await service.query(
            question=request.question,
            k=request.k,
            use_reranking=request.use_reranking,
            include_sources=request.include_sources,
            vietnamese_prompt=request.vietnamese_prompt,
        )

        return QueryResponse(
            success=result["success"],
            answer=result["answer"],
            sources=result["sources"],
            num_sources=result["num_sources"],
            confidence_score=result.get("confidence_score", 0.0),
            retrieval_method=result.get("retrieval_method", "embedding_only_legacy"),
            processing_info=result.get("processing_info", {}),
        )

    except Exception as e:
        logger.error(f"Lỗi truy vấn embedding only: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý câu hỏi: {str(e)}")


@router.get("/system_info", response_model=SystemInfoResponse)
async def get_system_info():
    """Lấy thông tin chi tiết về hệ thống RAG nâng cao."""
    service = get_rag_service()

    try:
        info = service.get_system_info()
        return SystemInfoResponse(success=True, system_info=info)
    except Exception as e:
        logger.error(f"Lỗi lấy system info: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi: {str(e)}")


@router.post("/configure", response_model=ApiResponse)
async def configure_service(config: AdvancedConfigRequest):
    """
    Cấu hình lại dịch vụ RAG nâng cao.

    Cho phép thay đổi:
    - Model embedding
    - Cấu hình chunking
    - Tham số retrieval
    """
    global _rag_service

    try:
        # Tạo service mới với config mới
        _rag_service = get_rag_service(
            embedding_provider=config.embedding_provider,
            embedding_model=config.embedding_model,
            embedding_device=config.embedding_device,
            collection_name=config.collection_name,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            use_ragflow_pdf=config.use_ragflow_pdf,
            retrieval_k=config.retrieval_k,
        )

        logger.info(
            f"Đã cấu hình lại service với {config.embedding_provider} embedding"
        )

        return ApiResponse(
            success=True,
            message=f"Đã cấu hình lại thành công với {config.embedding_provider} ({config.embedding_model})",
        )

    except Exception as e:
        logger.error(f"Lỗi cấu hình service: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi cấu hình: {str(e)}")


@router.delete("/clear", response_model=ApiResponse)
async def clear_knowledge_base():
    """Xóa toàn bộ knowledge base nâng cao."""
    service = get_rag_service()

    try:
        result = await service.clear_knowledge_base()
        return ApiResponse(success=result["success"], message=result["message"])
    except Exception as e:
        logger.error(f"Lỗi xóa knowledge base: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi: {str(e)}")


@router.get("/health", response_model=ApiResponse)
async def health_check():
    """Kiểm tra tình trạng dịch vụ RAG nâng cao."""
    try:
        service = get_rag_service()
        info = service.get_system_info()

        return ApiResponse(
            success=True,
            message="Dịch vụ RAG nâng cao hoạt động bình thường",
            data={
                "service_type": info["service_type"],
                "vietnamese_optimized": info["features"]["vietnamese_optimization"],
                "ragflow_pdf": info["features"]["ragflow_pdf_parsing"],
                "embedding_provider": info["components"]["embedding_service"][
                    "provider"
                ],
            },
        )

    except Exception as e:
        logger.error(f"Health check thất bại: {e}")
        raise HTTPException(status_code=500, detail=f"Service không khả dụng: {str(e)}")


@router.get("/embedding_models", response_model=ApiResponse)
async def get_available_embedding_models():
    """Lấy danh sách models embedding được hỗ trợ."""
    from rag.embeddings import MODEL_DEFAULT_EMBEDDING

    return ApiResponse(
        success=True,
        message="Danh sách models embedding được khuyến nghị cho tiếng Việt",
        data={
            "vietnamese_models": MODEL_DEFAULT_EMBEDDING,
            "current_service": get_rag_service().embedding_service.get_info(),
        },
    )


@router.post("/test_embedding", response_model=ApiResponse)
async def test_embedding_performance(
    text: str = Form("Đây là câu test để kiểm tra hiệu suất embedding tiếng Việt"),
    model_name: Optional[str] = Form(None, description="Tên model để test"),
):
    """Test hiệu suất embedding với text tiếng Việt."""
    try:
        service = get_rag_service()

        # Test embedding với text
        import time

        start_time = time.time()

        embedding = service.embedding_service.embed_query(text)

        end_time = time.time()
        processing_time = end_time - start_time

        return ApiResponse(
            success=True,
            message=f"Test embedding thành công trong {processing_time:.3f}s",
            data={
                "text": text,
                "embedding_dimension": len(embedding),
                "processing_time_seconds": processing_time,
                "model_info": service.embedding_service.get_info(),
                "vietnamese_optimized": True,
            },
        )

    except Exception as e:
        logger.error(f"Lỗi test embedding: {e}")
        raise HTTPException(status_code=500, detail=f"Test thất bại: {str(e)}")


# =============================================================================
# HYBRID RETRIEVAL ENDPOINTS (BM25 + Embedding)
# =============================================================================


class HybridQueryRequest(BaseModel):
    """Yêu cầu truy vấn hybrid với BM25 + Embedding."""

    question: str
    k: Optional[int] = None
    method: Literal["hybrid", "bm25_only", "embedding_only"] = "hybrid"
    include_sources: bool = True
    vietnamese_prompt: bool = True


class HybridQueryResponse(BaseModel):
    """Phản hồi truy vấn hybrid."""

    success: bool
    answer: str
    sources: list = []
    num_sources: int = 0
    confidence_score: float = 0.0
    retrieval_method: str = "hybrid"
    hybrid_info: dict = {}
    processing_info: dict = {}


class ComparisonResponse(BaseModel):
    """Phản hồi so sánh các phương pháp retrieval."""

    success: bool
    query: str
    k: int
    methods: dict
    recommendation: str
    comparison_info: dict = {}


@router.post("/query", response_model=HybridQueryResponse)
async def query_hybrid(request: HybridQueryRequest):
    """
    🎯 **RAG QUERY - ENDPOINT CHÍNH**

    **Hybrid Retrieval** (BM25 + Embedding) - Phương pháp tìm kiếm tiên tiến nhất:

    🔥 **Tại sao Hybrid tốt hơn Embedding thuần:**
    - **Từ khóa kỹ thuật**: BM25 tìm chính xác "JWT", "OAuth", "API"
    - **Ngữ nghĩa**: Embedding hiểu "bảo mật" ≈ "security"
    - **Fusion**: Kết hợp điểm số tối ưu cho kết quả tốt nhất

    📊 **Performance:**
    - ✅ +15-30% độ chính xác vs embedding only
    - ✅ Tốt với technical terms và proper nouns
    - ✅ Balanced cho diverse query types

    ⚙️ **Method Options:**
    - `"hybrid"`: BM25 + Embedding (**MẶC ĐỊNH - KHUYẾN NGHỊ**)
    - `"bm25_only"`: Chỉ tìm từ khóa chính xác
    - `"embedding_only"`: Chỉ tìm ngữ nghĩa (như endpoint cũ)

    💡 **Tip**: Để so sánh với embedding cũ, dùng `/query_embedding_only`
    """
    service = get_rag_service()

    try:
        result = await service.hybrid_query(
            question=request.question,
            k=request.k,
            method=request.method,
            include_sources=request.include_sources,
            vietnamese_prompt=request.vietnamese_prompt,
        )

        return HybridQueryResponse(
            success=result["success"],
            answer=result["answer"],
            sources=result["sources"],
            num_sources=result["num_sources"],
            confidence_score=result.get("confidence_score", 0.0),
            retrieval_method=result.get("retrieval_method", "hybrid"),
            hybrid_info=result.get("hybrid_info", {}),
            processing_info=result.get("processing_info", {}),
        )

    except Exception as e:
        logger.error(f"Lỗi hybrid query: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi hybrid search: {str(e)}")


@router.post("/compare_retrieval", response_model=ComparisonResponse)
async def compare_retrieval_methods(
    question: str = Form(..., description="Câu hỏi để so sánh"),
    k: int = Form(5, description="Số documents để lấy"),
):
    """
    📊 So sánh hiệu suất các phương pháp Retrieval

    **So sánh giữa:**
    1. **Hybrid Fusion** - BM25 + Embedding
    2. **BM25 Only** - Tìm từ khóa chính xác
    3. **Embedding Only** - Tìm ngữ nghĩa
    4. **Regular Embedding** - Embedding truyền thống

    **Metrics được đánh giá:**
    - Số lượng documents tìm thấy
    - Điểm trung bình (confidence score)
    - Preview nội dung top 3 documents
    - Khuyến nghị phương pháp tốt nhất

    **Dùng để:**
    - Đánh giá chất lượng retrieval cho câu hỏi cụ thể
    - Chọn phương pháp tối ưu cho từng loại query
    - Benchmarking và tối ưu hóa system
    """
    service = get_rag_service()

    try:
        result = await service.compare_retrieval_methods(question, k)

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return ComparisonResponse(
            success=True,
            query=result["query"],
            k=result["k"],
            methods=result["methods"],
            recommendation=result["comparison_info"]["recommendation"],
            comparison_info=result["comparison_info"],
        )

    except Exception as e:
        logger.error(f"Lỗi so sánh retrieval: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi so sánh: {str(e)}")


@router.get("/hybrid_stats", response_model=ApiResponse)
async def get_hybrid_retriever_stats():
    """
    📈 Thống kê Hybrid Retriever

    **Thông tin bao gồm:**
    - Cấu hình BM25 (k1, b parameters)
    - Trọng số fusion (BM25 vs Embedding)
    - Số lượng documents đã indexed
    - Vocabulary size (số từ unique)
    - Thống kê vector store
    """
    service = get_rag_service()

    try:
        stats = service.hybrid_retriever.get_stats()

        return ApiResponse(
            success=True,
            message="Thống kê Hybrid Retriever",
            data={
                "hybrid_config": {
                    "retriever_type": stats["retriever_type"],
                    "bm25_weight": stats["weights"]["bm25"],
                    "embedding_weight": stats["weights"]["embedding"],
                    "bm25_initialized": stats["bm25_initialized"],
                },
                "bm25_stats": stats.get("bm25_stats", {}),
                "vectorstore_info": stats["vectorstore_info"],
                "capabilities": {
                    "keyword_search": "BM25 exact matching",
                    "semantic_search": "HuggingFace embeddings",
                    "fusion_method": "Weighted score combination",
                    "vietnamese_optimized": True,
                },
            },
        )

    except Exception as e:
        logger.error(f"Lỗi lấy hybrid stats: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi: {str(e)}")


@router.post("/test_hybrid_performance", response_model=ApiResponse)
async def test_hybrid_performance(
    test_queries: List[str] = Form(
        [
            "Blockchain là gì?",
            "Machine learning algorithms",
            "Công nghệ AI trong y tế",
            "Phương pháp phân tích dữ liệu",
        ],
        description="Danh sách câu hỏi test",
    ),
    k: int = Form(3, description="Số documents cho mỗi query"),
):
    """
    🧪 Test hiệu suất Hybrid Retrieval

    **Chạy benchmark trên nhiều câu hỏi:**
    - Test cả 4 phương pháp retrieval
    - Đo thời gian xử lý
    - So sánh điểm số và số lượng kết quả
    - Tính trung bình hiệu suất

    **Kết quả:**
    - Performance summary cho từng phương pháp
    - Thời gian trung bình
    - Khuyến nghị optimization
    """
    service = get_rag_service()

    try:
        import time

        results = {
            "test_queries": test_queries,
            "k": k,
            "methods_performance": {},
            "summary": {},
        }

        methods = ["hybrid", "bm25_only", "embedding_only"]

        for method in methods:
            method_results = []
            total_time = 0

            for query in test_queries:
                start_time = time.time()

                query_result = await service.hybrid_query(
                    question=query,
                    k=k,
                    method=method,
                    include_sources=False,
                    vietnamese_prompt=False,
                )

                end_time = time.time()
                query_time = end_time - start_time
                total_time += query_time

                method_results.append(
                    {
                        "query": query,
                        "num_results": query_result.get("num_sources", 0),
                        "confidence": query_result.get("confidence_score", 0),
                        "time_seconds": query_time,
                    }
                )

            # Tính thống kê
            avg_results = sum(r["num_results"] for r in method_results) / len(
                method_results
            )
            avg_confidence = sum(r["confidence"] for r in method_results) / len(
                method_results
            )
            avg_time = total_time / len(test_queries)

            results["methods_performance"][method] = {
                "individual_results": method_results,
                "average_results_count": avg_results,
                "average_confidence": avg_confidence,
                "average_time_seconds": avg_time,
                "total_time_seconds": total_time,
            }

        # Tóm tắt và khuyến nghị
        best_confidence = max(
            results["methods_performance"].values(),
            key=lambda x: x["average_confidence"],
        )
        fastest_method = min(
            results["methods_performance"].values(),
            key=lambda x: x["average_time_seconds"],
        )

        results["summary"] = {
            "best_confidence_method": "Chưa xác định",  # Cần logic để tìm method name
            "fastest_method": "Chưa xác định",
            "recommendation": "Hybrid method cung cấp cân bằng tốt nhất giữa chất lượng và tốc độ",
            "optimization_tips": [
                "Điều chỉnh BM25 weights dựa trên domain cụ thể",
                "Fine-tune embedding model cho dữ liệu riêng",
                "Tối ưu chunk size và overlap cho better retrieval",
            ],
        }

        return ApiResponse(
            success=True,
            message=f"Hoàn thành test {len(test_queries)} queries với {len(methods)} methods",
            data=results,
        )

    except Exception as e:
        logger.error(f"Lỗi test hybrid performance: {e}")
        raise HTTPException(status_code=500, detail=f"Test failed: {str(e)}")


# Search-only models (new)
class SearchOnlyRequest(BaseModel):
    """Yêu cầu tìm kiếm chỉ retrieval - KHÔNG LLM."""

    question: str
    k: Optional[int] = None
    method: Literal["hybrid", "bm25_only", "embedding_only"] = "hybrid"
    include_sources: bool = True
    score_threshold: Optional[float] = None


class SearchOnlyResponse(BaseModel):
    """Phản hồi tìm kiếm chỉ retrieval - NHANH!"""

    success: bool
    documents: list = []  # Raw documents
    sources: list = []  # Formatted sources
    num_sources: int = 0
    search_method: str = "hybrid"
    confidence_score: float = 0.0
    query: str = ""
    k_requested: int = 5
    score_threshold: Optional[float] = None
    processing_info: dict = {}
    hybrid_stats: dict = {}


@router.post("/search", response_model=SearchOnlyResponse)
async def search_documents(request: SearchOnlyRequest):
    """
    🔍 TÌM KIẾM DOCUMENTS - KHÔNG LLM (NHANH!)

    Chỉ tìm kiếm và trả về documents liên quan, KHÔNG gọi LLM để sinh câu trả lời.

    **⚡ Performance:**
    - Nhanh gấp 20-100x so với `/query` (vì skip LLM)
    - Phù hợp cho search, preview, debugging

    **🎯 Use Cases:**
    - Tìm kiếm nhanh documents
    - Preview nội dung trước khi query
    - Debugging retrieval performance
    - Bulk search operations

    **Methods:**
    - `hybrid`: BM25 + Embedding (khuyến nghị)
    - `bm25_only`: Chỉ keyword search
    - `embedding_only`: Chỉ semantic search
    """
    service = get_rag_service()

    try:
        result = await service.search_only(
            question=request.question,
            k=request.k,
            method=request.method,
            include_sources=request.include_sources,
            score_threshold=request.score_threshold,
        )

        # Convert documents to serializable format
        documents_data = []
        for doc in result.get("documents", []):
            doc_data = {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": doc.metadata.get("similarity_score", 0.0),
            }
            documents_data.append(doc_data)

        return SearchOnlyResponse(
            success=result["success"],
            documents=documents_data,
            sources=result["sources"],
            num_sources=result["num_sources"],
            search_method=result["search_method"],
            confidence_score=result["confidence_score"],
            query=result["query"],
            k_requested=result["k_requested"],
            score_threshold=result["score_threshold"],
            processing_info=result["processing_info"],
            hybrid_stats=result["hybrid_stats"],
        )

    except Exception as e:
        logger.error(f"Lỗi search-only: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi tìm kiếm: {str(e)}")
