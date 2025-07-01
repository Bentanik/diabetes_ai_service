"""Models cho RAG API - Request và Response schemas."""

from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class FileInfoModel(BaseModel):
    """Model cho thông tin file."""

    filename: str = Field(description="Tên file")
    file_size: int = Field(description="Kích thước file (bytes)")
    file_extension: str = Field(description="Extension của file")
    content_type: str = Field(description="MIME type của file")
    upload_time: str = Field(description="Thời gian upload")
    storage_path: Optional[str] = Field(description="Đường dẫn lưu trữ trong MinIO")
    storage_time: Optional[str] = Field(description="Thời gian lưu trữ trong MinIO")


class FileUploadResponse(BaseModel):
    """Response model cho file upload và processing."""

    success: bool = Field(description="Trạng thái thành công")
    message: str = Field(description="Thông điệp mô tả kết quả")
    file_info: FileInfoModel = Field(description="Thông tin file đã upload")
    document_ids: List[str] = Field(
        description="Danh sách document_ids đã được process"
    )
    statistics: Dict[str, Any] = Field(description="Thống kê processing")
    processing_time: float = Field(description="Thời gian xử lý (giây)")


class DocumentInfo(BaseModel):
    """Model cho thông tin chi tiết của một document."""

    filename: str = Field(description="Tên file")
    size: int = Field(description="Kích thước file (bytes)")
    last_modified: Optional[str] = Field(description="Thời gian chỉnh sửa cuối")
    content_type: str = Field(description="Loại file")


class CollectionStats(BaseModel):
    """Model cho thống kê documents trong một collection."""

    total_documents: int = Field(description="Tổng số documents")
    total_size_bytes: int = Field(description="Tổng dung lượng (bytes)")
    file_types: Dict[str, int] = Field(description="Số lượng file theo loại")
    documents: List[DocumentInfo] = Field(
        description="Danh sách chi tiết các documents"
    )
    collection_name: str = Field(description="Tên collection")
    last_updated: str = Field(description="Thời gian cập nhật thống kê")


class ChunkModel(BaseModel):
    """Model cho một chunk."""

    content: str = Field(description="Nội dung chunk")
    chunk_id: int = Field(description="ID của chunk")
    global_chunk_id: str = Field(description="Global unique ID của chunk")
    content_length: int = Field(description="Độ dài nội dung")
    metadata: Dict[str, Any] = Field(description="Metadata của chunk")


class ProcessingStats(BaseModel):
    """Model cho thống kê processing."""

    total_files_processed: int = Field(description="Tổng số file đã xử lý")
    total_documents_created: int = Field(description="Tổng số document đã tạo")
    total_chunks_created: int = Field(description="Tổng số chunk đã tạo")
    processing_errors: int = Field(description="Số lỗi processing")
    success_rate: float = Field(description="Tỷ lệ thành công (%)")
    avg_chunks_per_file: float = Field(description="Trung bình chunk trên file")
    last_processing_time: Optional[str] = Field(description="Thời gian processing cuối")


class ErrorResponse(BaseModel):
    """Response model cho lỗi."""

    success: bool = False
    error: str = Field(description="Thông điệp lỗi")
    error_code: str = Field(description="Mã lỗi")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Chi tiết lỗi")


class KnowledgeBaseCreate(BaseModel):
    """Schema cho request tạo knowledge base."""

    name: str = Field(
        ..., description="Tên của knowledge base", min_length=1, max_length=255
    )
    description: Optional[str] = Field(None, description="Mô tả về knowledge base")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata bổ sung")


class KnowledgeBaseUpdate(BaseModel):
    """Schema cho request cập nhật knowledge base."""

    name: Optional[str] = Field(
        None, description="Tên mới của knowledge base", min_length=1, max_length=255
    )
    description: Optional[str] = Field(None, description="Mô tả mới về knowledge base")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata mới")


class KnowledgeBaseResponse(BaseModel):
    """Schema cho response knowledge base."""

    name: str = Field(description="Tên của knowledge base")
    description: Optional[str] = Field(None, description="Mô tả về knowledge base")
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Metadata của knowledge base"
    )
    collection_info: Dict[str, Any] = Field(description="Thông tin về collection")
    created_at: str = Field(description="Thời gian tạo")

    class Config:
        from_attributes = True


class KnowledgeBaseList(BaseModel):
    """Schema cho danh sách knowledge bases."""

    knowledge_bases: List[KnowledgeBaseResponse] = Field(
        description="Danh sách knowledge bases"
    )
    total: int = Field(description="Tổng số knowledge bases")


class MultiCollectionSearchRequest(BaseModel):
    """Model cho yêu cầu tìm kiếm trên nhiều knowledge bases."""

    query: str = Field(description="Câu hỏi/query cần tìm kiếm")
    collection_names: List[str] = Field(
        description="Danh sách các knowledge bases cần tìm kiếm"
    )
    top_k: Optional[int] = Field(default=5, description="Số lượng kết quả trả về")
    score_threshold: Optional[float] = Field(
        default=0.3, description="Ngưỡng điểm tối thiểu (0-1)", ge=0.0, le=1.0
    )


class SearchResult(BaseModel):
    """Model cho một kết quả tìm kiếm."""

    content: str = Field(description="Nội dung đoạn văn bản")
    metadata: Dict[str, Any] = Field(description="Metadata của đoạn văn bản")
    score: float = Field(description="Điểm tương đồng")
    collection_name: str = Field(description="Knowledge base chứa kết quả này")


class MultiCollectionSearchResponse(BaseModel):
    """Model cho kết quả tìm kiếm từ nhiều knowledge bases."""

    results: List[SearchResult] = Field(description="Danh sách kết quả tìm kiếm")
    total_results: int = Field(description="Tổng số kết quả tìm được")
    processing_time: float = Field(description="Thời gian xử lý (giây)")
    collection_stats: Dict[str, Any] = Field(
        description="Thống kê theo từng collection"
    )
