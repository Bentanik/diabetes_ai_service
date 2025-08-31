from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from core.cqrs import Mediator
from utils import get_logger
from app.feature.user import CreateUserProfileCommand, CreateHealthRecordCommand

router = APIRouter(prefix="/api/v1/users", tags=["Users"])
logger = get_logger(__name__)


@router.post(
    "",
    summary="Tạo hồ sơ người dùng",
    description="Tạo hồ sơ người dùng",
)
async def create_user_profile(req: CreateUserProfileCommand) -> JSONResponse:
    """
    Endpoint tạo hồ sơ người dùng.

    Args:
        req (CreateUserProfileCommand): Command chứa thông tin hồ sơ người dùng cần tạo

    Returns:
        JSONResponse
    """
    logger.info(f"Tạo hồ sơ người dùng: {req.user_id}")
    try:
        result = await Mediator.send(req)
        return result.to_response()
    except Exception as e:
        logger.error(f"Lỗi khi tạo hồ sơ người dùng: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post(
    "/health-records",
    summary="Tạo dữ liệu sức khỏe người dùng",
    description="Tạo dữ liệu sức khỏe người dùng",
)
async def create_health_record(req: CreateHealthRecordCommand) -> JSONResponse:
    """
    Endpoint tạo dữ liệu sức khỏe người dùng.

    Args:
        req (CreateHealthRecordCommand): Command chứa thông tin dữ liệu sức khỏe người dùng cần tạo

    Returns:
        JSONResponse
    """ 
    logger.info(f"Tạo dữ liệu sức khỏe người dùng: {req.user_id}")
    try:
        result = await Mediator.send(req)
        return result.to_response()
    except Exception as e:
        logger.error(f"Lỗi khi tạo dữ liệu sức khỏe người dùng: {e}")
        raise HTTPException(status_code=500, detail=str(e))