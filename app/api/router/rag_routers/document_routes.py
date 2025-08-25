"""
Document Routes - Module định nghĩa các API endpoints cho quản lý tài liệu

File này cung cấp các REST API endpoints để thực hiện các thao tác CRUD
(Create, Read, Update, Delete) trên tài liệu:
- POST /documents: Tạo mới tài liệu
- GET /documents: Lấy danh sách có phân trang và tìm kiếm
- GET /documents/{id}: Lấy chi tiết một tài liệu
- PUT /documents/{id}: Cập nhật thông tin tài liệu
- DELETE /documents/{id}: Xóa tài liệu
- GET /documents/{id}/download: Tải file tài liệu
"""

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from app.feature.document import (
    CreateDocumentCommand,
    GetDocumentsQuery,
    GetDocumentQuery,
    GetDocumentChunksQuery,
    DeleteDocumentCommand,
    UpdateDocumentCommand,
)
from app.feature.document.commands.change_document_status_command import ChangeDocumentChunkStatusCommand
from core.cqrs import Mediator
from utils import get_logger, should_compress, compress_stream, get_best_compression
from typing import cast
from app.dto.models import DocumentModelDTO
from app.storage import MinioManager
import mimetypes
import urllib

# Khởi tạo router với prefix và tag
router = APIRouter(prefix="/documents", tags=["Documents"])
logger = get_logger(__name__)


@router.post(
    "",
    response_model=None,
    summary="Tạo tài liệu mới",
    description="Tạo mới tài liệu trong hệ thống với file upload.",
)
async def create_document(
    file: UploadFile = File(..., description="File tài liệu cần upload"),
    knowledge_id: str = Form(..., description="ID của cơ sở tri thức chứa tài liệu"),
    title: str = Form(..., description="Tiêu đề tài liệu"),
    description: str = Form(..., description="Mô tả về tài liệu"),
) -> JSONResponse:
    """
    Endpoint tạo mới tài liệu.

    Args:
        file (UploadFile): File tài liệu cần upload
        knowledge_id (str): ID của cơ sở tri thức
        title (str): Tiêu đề tài liệu
        description (str): Mô tả tài liệu

    Returns:
        JSONResponse

    Raises:
        HTTPException: Khi có lỗi xảy ra trong quá trình xử lý
    """
    logger.info(f"Tạo tài liệu mới: {title}")
    try:
        command = CreateDocumentCommand(
            file=file, knowledge_id=knowledge_id, title=title, description=description
        )
        result = await Mediator.send(command)
        return result.to_response()
    except Exception as e:
        logger.error(f"Lỗi tạo tài liệu: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Tạo tài liệu thất bại")

@router.get(
    "",
    response_model=None,
    summary="Lấy danh sách tài liệu",
    description="Lấy danh sách tài liệu với tìm kiếm, phân trang và sắp xếp.",
)
async def get_documents(
    knowledge_id: str = Query(None, description="ID của cơ sở tri thức để filter"),
    search: str = Query("", description="Tiêu đề tài liệu cần tìm kiếm"),
    page: int = Query(1, ge=1, description="Trang hiện tại"),
    limit: int = Query(10, ge=1, le=100, description="Số lượng bản ghi mỗi trang"),
    sort_by: str = Query("updated_at", description="Trường cần sắp xếp"),
    sort_order: str = Query(
        "desc",
        pattern="^(asc|desc)$",
        description="Thứ tự sắp xếp: asc hoặc desc",
    ),
) -> JSONResponse:
    """
    Endpoint lấy danh sách tài liệu có phân trang.

    Args:
        knowledge_id (str): ID cơ sở tri thức để filter (optional)
        search (str): Từ khóa tìm kiếm theo tiêu đề
        page (int): Số trang, bắt đầu từ 1
        limit (int): Số bản ghi mỗi trang (1-100)
        sort_by (str): Trường dùng để sắp xếp
        sort_order (str): Thứ tự sắp xếp (asc/desc)

    Returns:
        JSONResponse

    Raises:
        HTTPException: Khi có lỗi xảy ra trong quá trình xử lý
    """
    logger.info(f"Lấy danh sách tài liệu - search={search}, page={page}")
    try:
        query = GetDocumentsQuery(
            knowledge_id=knowledge_id,
            search=search,
            page=page,
            limit=limit,
            sort_by=sort_by,
            sort_order=sort_order,
        )
        result = await Mediator.send(query)
        return result.to_response()
    except Exception as e:
        logger.error(f"Lỗi lấy danh sách tài liệu: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Không thể lấy danh sách tài liệu")


@router.get(
    "/{document_id}",
    response_model=None,
    summary="Lấy thông tin tài liệu",
    description="Lấy thông tin chi tiết của một tài liệu theo ID.",
)
async def get_document(document_id: str) -> JSONResponse:
    """
    Endpoint lấy thông tin chi tiết một tài liệu.

    Args:
        document_id (str): ID của tài liệu cần lấy thông tin

    Returns:
        JSONResponse

    Raises:
        HTTPException: Khi có lỗi xảy ra trong quá trình xử lý
    """
    logger.info(f"Lấy thông tin tài liệu: {document_id}")
    try:
        query = GetDocumentQuery(id=document_id)
        result = await Mediator.send(query)
        return result.to_response()
    except Exception as e:
        logger.error(f"Lỗi lấy thông tin tài liệu: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Không thể lấy thông tin tài liệu")

@router.get(
    "/{document_id}/chunks",
    response_model=None,
    summary="Lấy thông tin phân tích tài liệu",
    description="Lấy thông tin phân tích tài liệu theo ID.",
)
async def get_document_chunks(
    document_id: str,
    min_diabetes_score: float = Query(None, description="Điểm diabetes tối thiểu"),
    max_diabetes_score: float = Query(None, description="Điểm diabetes tối đa"),
    page: int = Query(1, ge=1, description="Trang hiện tại"),
    limit: int = Query(10, ge=1, le=100, description="Số lượng bản ghi mỗi trang"),
    sort_by: str = Query("updated_at", description="Trường cần sắp xếp"),
    sort_order: str = Query(
        "desc",
        pattern="^(asc|desc)$",
        description="Thứ tự sắp xếp: asc hoặc desc",
    ),
) -> JSONResponse:
    """
    Endpoint lấy thông tin chi tiết một tài liệu.

    Args:
        document_id (str): ID của tài liệu cần lấy thông tin

    Returns:
        JSONResponse

    Raises:
        HTTPException: Khi có lỗi xảy ra trong quá trình xử lý
    """
    logger.info(f"Lấy thông tin tài liệu: {document_id}")
    try:
        query = GetDocumentChunksQuery(document_id=document_id, page=page, limit=limit, sort_by=sort_by, sort_order=sort_order, min_diabetes_score=min_diabetes_score, max_diabetes_score=max_diabetes_score)
        result = await Mediator.send(query)
        return result.to_response()
    except Exception as e:
        logger.error(f"Lỗi lấy thông tin tài liệu: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Không thể lấy thông tin tài liệu")

@router.put(
    "",
    response_model=None,
    summary="Cập nhật trạng thái tài liệu",
    description="Cập nhật trạng thái tài liệu theo ID.",
)

async def update_document(request: UpdateDocumentCommand) -> JSONResponse:
    """
    Endpoint cập nhật trạng thái tài liệu.

    Args:
        document_id (str): ID của tài liệu cần cập nhật trạng thái

    Returns:
        JSONResponse

    Raises:
        HTTPException: Khi có lỗi xảy ra trong quá trình xử lý
    """
    try:
        result = await Mediator.send(request)
        return result.to_response()
    except Exception as e:
        logger.error(f"Lỗi cập nhật trạng thái tài liệu: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Cập nhật trạng thái tài liệu thất bại")

@router.delete(
    "/{document_id}",
    response_model=None,
    summary="Xóa tài liệu",
    description="Xóa một tài liệu khỏi hệ thống theo ID.",
)
async def delete_document(document_id: str) -> JSONResponse:
    """
    Endpoint xóa tài liệu.

    Args:
        document_id (str): ID của tài liệu cần xóa

    Returns:
        JSONResponse

    Raises:
        HTTPException: Khi có lỗi xảy ra trong quá trình xử lý
    """
    logger.info(f"Xóa tài liệu: {document_id}")
    try:
        command = DeleteDocumentCommand(id=document_id)
        result = await Mediator.send(command)
        return result.to_response()
    except Exception as e:
        logger.error(f"Lỗi xóa tài liệu: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Xóa tài liệu thất bại")

@router.put(
    "/chunks/change-status",
    response_model=None,
    summary="Cập nhật trạng thái tài liệu",
    description="Cập nhật trạng thái tài liệu theo ID.",
)

async def update_document_status(request: ChangeDocumentChunkStatusCommand) -> JSONResponse:
    """
    Endpoint cập nhật trạng thái tài liệu.

    Args:
        document_id (str): ID của tài liệu cần cập nhật trạng thái

    Returns:
        JSONResponse

    Raises:
        HTTPException: Khi có lỗi xảy ra trong quá trình xử lý
    """
    try:
        result = await Mediator.send(request)
        return result.to_response()
    except Exception as e:
        logger.error(f"Lỗi cập nhật trạng thái tài liệu: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Cập nhật trạng thái tài liệu thất bại")

@router.get(
    "/{document_id}/download",
    summary="Tải tài liệu",
    description="Tải file tài liệu từ storage với hỗ trợ compression.",
)
async def download_document(
    document_id: str,
    request: Request,
    compress: bool = Query(False, description="Có nén file không"),
    compression_type: str = Query("gzip", description="Loại nén (gzip, deflate)"),
    chunk_size: int = Query(
        64 * 1024, ge=8192, le=1024 * 1024, description="Kích thước chunk"
    ),
):
    logger.info(f"Tải tài liệu: {document_id}")
    try:
        # Lấy thông tin document
        query = GetDocumentQuery(id=document_id)
        result = await Mediator.send(query)

        if not result.is_success:
            return result.to_response()

        document_dto = cast(DocumentModelDTO, result.data)

        # Parse file path
        file_path_parts = document_dto.file.path.split("/", 1)
        bucket_name = file_path_parts[0]
        object_name = file_path_parts[1] if len(file_path_parts) > 1 else ""

        # Lấy stream từ MinIO
        stream_info = await MinioManager.get_instance().get_stream_async(
            bucket_name, object_name, chunk_size
        )

        filename = document_dto.title or stream_info["filename"]
        file_size = stream_info["size"]

        # Nếu filename không có phần mở rộng, lấy từ stream_info["filename"]
        if "." not in filename and "." in stream_info["filename"]:
            filename = stream_info["filename"]

        # Xác định MIME type
        mime_type, _ = mimetypes.guess_type(filename)
        if not mime_type:
            mime_type = "application/octet-stream"

        # Compression
        compression_method = None
        if compress and should_compress(filename, file_size):
            accept_encoding = request.headers.get("Accept-Encoding", "")
            compression_method = get_best_compression(accept_encoding, compression_type)

        # Headers
        if compression_method:
            processed_stream = compress_stream(
                stream_info["stream"], compression_method
            )
            headers = {
                "Content-Disposition": f"attachment; filename*=UTF-8''{urllib.parse.quote(filename)}",
                "Content-Encoding": compression_method,
                "Transfer-Encoding": "chunked",
                "X-Original-Size": str(file_size),
            }
        else:
            processed_stream = stream_info["stream"]
            headers = {
                "Content-Disposition": f"attachment; filename*=UTF-8''{urllib.parse.quote(filename)}",
                "Content-Length": str(file_size),
            }

        headers.update(
            {
                "Content-Type": mime_type,
                "Cache-Control": "public, max-age=3600",
            }
        )

        return StreamingResponse(processed_stream, headers=headers)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Lỗi tải tài liệu: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Tải tài liệu thất bại")