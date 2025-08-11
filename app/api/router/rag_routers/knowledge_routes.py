"""
Knowledge Routes - Module định nghĩa các API endpoints cho quản lý cơ sở tri thức

File này cung cấp các REST API endpoints để thực hiện các thao tác CRUD
(Create, Read, Update, Delete) trên cơ sở tri thức:
- POST /knowledges: Tạo mới cơ sở tri thức
- GET /knowledges: Lấy danh sách có phân trang và tìm kiếm
- GET /knowledges/{id}: Lấy chi tiết một cơ sở tri thức
- PUT /knowledges/{id}: Cập nhật thông tin cơ sở tri thức
- DELETE /knowledges/{id}: Xóa cơ sở tri thức
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from app.api.schemas import UpdateKnowledgeRequest
from core.cqrs import Mediator
from utils import get_logger
from app.feature.knowledge import (
    CreateKnowledgeCommand,
    UpdateKnowledgeCommand,
    DeleteKnowledgeCommand,
    GetKnowledgesQuery,
    GetKnowledgeQuery,
)

# Khởi tạo router với prefix và tag
router = APIRouter(prefix="/knowledges", tags=["Knowledges"])
logger = get_logger(__name__)


@router.post(
    "",
    response_model=None,
    summary="Tạo cơ sở tri thức mới",
    description="Tạo một cơ sở tri thức mới trong hệ thống.",
)
async def create_knowledge(kb_req: CreateKnowledgeCommand) -> JSONResponse:
    """
    Endpoint tạo mới cơ sở tri thức.

    Args:
        kb_req (CreateKnowledgeCommand): Command chứa thông tin cơ sở tri thức cần tạo

    Returns:
        JSONResponse
    Raises:
        HTTPException: Khi có lỗi xảy ra trong quá trình xử lý
    """
    logger.info(f"Tạo cơ sở tri thức mới: {kb_req.name}")
    try:
        result = await Mediator.send(kb_req)
        return result.to_response()
    except Exception as e:
        logger.error(f"Lỗi tạo cơ sở tri thức: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Tạo cơ sở tri thức thất bại")


@router.get(
    "",
    response_model=None,
    summary="Lấy danh sách cơ sở tri thức",
    description="Lấy danh sách cơ sở tri thức với tìm kiếm, phân trang và sắp xếp.",
)
async def get_knowledges(
    search: str = Query("", description="Tên cơ sở tri thức cần tìm kiếm"),
    select_training: Optional[bool] = Query(None, description="Có được chọn để huấn luyện hay không"),
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
    Endpoint lấy danh sách cơ sở tri thức có phân trang.

    Args:
        search (str): Từ khóa tìm kiếm theo tên
        page (int): Số trang, bắt đầu từ 1
        limit (int): Số bản ghi mỗi trang (1-100)
        sort_by (str): Trường dùng để sắp xếp
        sort_order (str): Thứ tự sắp xếp (asc/desc)

    Returns:
        JSONResponse

    Raises:
        HTTPException: Khi có lỗi xảy ra trong quá trình xử lý
    """
    logger.info(f"Lấy danh sách cơ sở tri thức - search={search}, page={page}")
    try:
        query = GetKnowledgesQuery(
            search=search,
            page=page,
            limit=limit,
            sort_by=sort_by,
            sort_order=sort_order,
            select_training=select_training,
        )
        result = await Mediator.send(query)
        return result.to_response()
    except Exception as e:
        logger.error(f"Lỗi lấy danh sách cơ sở tri thức: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Không thể lấy danh sách cơ sở tri thức"
        )


@router.put(
    "/{id}",
    response_model=None,
    summary="Cập nhật cơ sở tri thức",
    description="Cập nhật thông tin của một cơ sở tri thức theo ID.",
)
async def update_knowledge(id: str, req: UpdateKnowledgeRequest) -> JSONResponse:
    """
    Endpoint cập nhật thông tin cơ sở tri thức.

    Args:
        id (str): ID của cơ sở tri thức cần cập nhật
        req (UpdateKnowledgeRequest): Dữ liệu cập nhật

    Returns:
        JSONResponse
    Raises:
        HTTPException: Khi có lỗi xảy ra trong quá trình xử lý
    """
    logger.info(f"Cập nhật cơ sở tri thức: id={id}")
    try:
        command = UpdateKnowledgeCommand(
            id=id,
            name=req.name,
            description=req.description,
            select_training=req.select_training,
        )
        result = await Mediator.send(command)
        return result.to_response()
    except Exception as e:
        logger.error(f"Lỗi cập nhật cơ sở tri thức: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Cập nhật cơ sở tri thức thất bại")


@router.delete(
    "/{id}",
    response_model=None,
    summary="Xóa cơ sở tri thức",
    description="Xóa một cơ sở tri thức khỏi hệ thống theo ID.",
)
async def delete_knowledge(id: str) -> JSONResponse:
    """
    Endpoint xóa cơ sở tri thức.

    Args:
        id (str): ID của cơ sở tri thức cần xóa

    Returns:
        JSONResponse

    Raises:
        HTTPException: Khi có lỗi xảy ra trong quá trình xử lý
    """
    logger.info(f"Xóa cơ sở tri thức: id={id}")
    try:
        command = DeleteKnowledgeCommand(id=id)
        result = await Mediator.send(command)
        return result.to_response()
    except Exception as e:
        logger.error(f"Lỗi xóa cơ sở tri thức: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Xóa cơ sở tri thức thất bại")


@router.get(
    "/{id}",
    response_model=None,
    summary="Lấy cơ sở tri thức",
    description="Lấy thông tin chi tiết của một cơ sở tri thức theo ID.",
)
async def get_knowledge(id: str) -> JSONResponse:
    """
    Endpoint lấy thông tin chi tiết một cơ sở tri thức.

    Args:
        id (str): ID của cơ sở tri thức cần lấy thông tin

    Returns:
        JSONResponse

    Raises:
        HTTPException: Khi có lỗi xảy ra trong quá trình xử lý
    """
    logger.info(f"Lấy cơ sở tri thức: id={id}")
    try:
        query = GetKnowledgeQuery(id=id)
        result = await Mediator.send(query)
        return result.to_response()
    except Exception as e:
        logger.error(f"Lỗi lấy cơ sở tri thức: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Không thể lấy cơ sở tri thức")
