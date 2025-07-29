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

import urllib
from typing import cast
from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from app.feature.document import (
    CreateDocumentCommand,
    GetDocumentQuery,
    GetDocumentsQuery,
    UpdateDocumentCommand,
    DeleteDocumentCommand,
)
from app.dto.models import DocumentModelDTO
from app.storage import MinioManager
from core.cqrs import Mediator
from shared.messages.document_message import DocumentResult
from utils import (
    compress_stream,
    get_best_compression,
    get_logger,
    should_compress,
)

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


@router.put(
    "/{document_id}",
    response_model=None,
    summary="Cập nhật tài liệu",
    description="Cập nhật thông tin của một tài liệu theo ID.",
)
async def update_document(
    document_id: str,
    title: str = Form(None, description="Tiêu đề mới của tài liệu"),
    description: str = Form(None, description="Mô tả mới của tài liệu"),
    priority_diabetes: float = Form(
        None, description="Độ ưu tiên liên quan đến tiểu đường (0.0-1.0)"
    ),
) -> JSONResponse:
    """
    Endpoint cập nhật thông tin tài liệu.

    Args:
        document_id (str): ID của tài liệu cần cập nhật
        title (str): Tiêu đề mới (optional)
        description (str): Mô tả mới (optional)
        priority_diabetes (float): Độ ưu tiên tiểu đường (optional)

    Returns:
        JSONResponse

    Raises:
        HTTPException: Khi có lỗi xảy ra trong quá trình xử lý
    """
    logger.info(f"Cập nhật tài liệu: {document_id}")
    try:
        command = UpdateDocumentCommand(
            id=document_id,
            title=title,
            description=description,
            priority_diabetes=priority_diabetes,
        )
        result = await Mediator.send(command)
        return result.to_response()
    except Exception as e:
        logger.error(f"Lỗi cập nhật tài liệu: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Cập nhật tài liệu thất bại")


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
    """
    Endpoint tải file tài liệu.

    Args:
        document_id (str): ID của tài liệu cần tải
        request (Request): HTTP request object
        compress (bool): Có nén file không
        compression_type (str): Loại nén
        chunk_size (int): Kích thước chunk để streaming

    Returns:
        StreamingResponse: File stream

    Raises:
        HTTPException: Khi có lỗi xảy ra trong quá trình xử lý
    """
    logger.info(f"Tải tài liệu: {document_id}")
    try:
        # Sử dụng Mediator để lấy thông tin document
        query = GetDocumentQuery(id=document_id)
        result = await Mediator.send(query)

        if not result.is_success:
            return result.to_response()

        document_dto = cast(DocumentModelDTO, result.data)

        # Parse file path từ storage
        file_path_parts = document_dto.file.path.split("/", 1)
        bucket_name = file_path_parts[0]
        object_name = file_path_parts[1] if len(file_path_parts) > 1 else ""

        # Lấy stream từ MinIO
        stream_info = await MinioManager.get_instance().get_stream_async(
            bucket_name, object_name, chunk_size
        )

        filename = document_dto.title or stream_info["filename"]
        file_size = stream_info["size"]

        # Xử lý compression nếu cần
        compression_method = None
        if compress and should_compress(filename, file_size):
            accept_encoding = request.headers.get("Accept-Encoding", "")
            compression_method = get_best_compression(accept_encoding, compression_type)

        # Chuẩn bị headers và stream
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
                "Content-Type": "application/octet-stream",
                "Cache-Control": "public, max-age=3600",
            }
        )

        return StreamingResponse(processed_stream, headers=headers)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Lỗi tải tài liệu: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Tải tài liệu thất bại")
