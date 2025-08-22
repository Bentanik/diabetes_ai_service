from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from app.feature.settings.commands import UpdateSettingCommand
from app.feature.settings.query import GetSettingQuery
from core.cqrs import Mediator
from utils import (
    get_logger,
)

# Khởi tạo router với prefix và tag
router = APIRouter(prefix="/setting", tags=["Setting"])
logger = get_logger(__name__)

@router.put(
    "",
    response_model=None,
    summary="Cập nhật cài đặt",
    description="Cập nhật cài đặt.",
)
async def update_setting(req: UpdateSettingCommand) -> JSONResponse:
    """
    Endpoint cập nhật cài đặt.

    Args:
        req (AddTrainingDocumentCommand): Command chứa thông tin tài liệu cần thêm

    Returns:
        JSONResponse
    Raises:
        HTTPException: Khi có lỗi xảy ra trong quá trình xử lý
    """

    try:
        result = await Mediator.send(req)
        return result.to_response()
    except Exception as e:
        logger.error(f"Lỗi khi thêm tài liệu vào vector database: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "",
    response_model=None,
    summary="Lấy cài đặt",
    description="Lấy cài đặt.",
)
async def get_setting() -> JSONResponse:
    try:
        result = await Mediator.send(GetSettingQuery())
        return result.to_response()
    except Exception as e:
        logger.error(f"Lỗi khi lấy cài đặt: {e}")
        raise HTTPException(status_code=500, detail=str(e))
