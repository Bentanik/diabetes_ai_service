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
