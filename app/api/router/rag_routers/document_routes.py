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

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from app.feature.document import (
    CreateDocumentCommand,
    GetDocumentsQuery,
    GetDocumentQuery,
    GetDocumentChunksQuery,
)
from core.cqrs import Mediator
from utils import get_logger

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
        query = GetDocumentChunksQuery(document_id=document_id, page=page, limit=limit, sort_by=sort_by, sort_order=sort_order)
        result = await Mediator.send(query)
        return result.to_response()
    except Exception as e:
        logger.error(f"Lỗi lấy thông tin tài liệu: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Không thể lấy thông tin tài liệu")