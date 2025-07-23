from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse

from app.api.schemas import UpdateKnowledgeRequest
from utils import get_logger
from core.cqrs import Mediator
from app.feature.knowledge import (
    CreateKnowledgeCommand,
    UpdateKnowledgeCommand,
    DeleteKnowledgeCommand,
    GetKnowledgesQuery,
    GetKnowledgeQuery,
)

router = APIRouter(prefix="/knowledge", tags=["Knowledge"])
logger = get_logger(__name__)


@router.post(
    "",
    response_model=None,
    summary="Tạo cơ sở tri thức mới",
    description="Tạo một cơ sở tri thức mới trong hệ thống.",
)
async def create_knowledge(kb_req: CreateKnowledgeCommand) -> JSONResponse:
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
    search_name: str = Query("", description="Tên cơ sở tri thức cần tìm kiếm"),
    page: int = Query(1, ge=1, description="Trang hiện tại"),
    limit: int = Query(10, ge=1, le=100, description="Số lượng bản ghi mỗi trang"),
    sort_by: str = Query("updated_at", description="Trường cần sắp xếp"),
    sort_order: str = Query(
        "desc",
        pattern="^(asc|desc)$",
        description="Thứ tự sắp xếp: asc hoặc desc",
    ),
) -> JSONResponse:
    logger.info(
        f"Lấy danh sách cơ sở tri thức - search_name={search_name}, page={page}"
    )
    try:
        query = GetKnowledgesQuery(
            search_name=search_name,
            page=page,
            limit=limit,
            sort_by=sort_by,
            sort_order=sort_order,
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
    description="Lấy cơ sở tri thức.",
)
async def get_knowledge(
    id: str,
) -> JSONResponse:
    logger.info(f"Lấy cơ sở tri thức: id={id}")
    try:
        query = GetKnowledgeQuery(id=id)
        result = await Mediator.send(query)
        return result.to_response()
    except Exception as e:
        logger.error(f"Lỗi lấy danh sách cơ sở tri thức: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Không thể lấy danh sách cơ sở tri thức"
        )
