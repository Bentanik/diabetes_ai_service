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

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from app.feature.document import (
    CreateDocumentCommand,
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