from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from app.feature.chat import (
    CreateChatCommand,
    GetChatHistoriesQuery,
)
from core.cqrs import Mediator
from utils import (
    get_logger,
)

# Khởi tạo router với prefix và tag
router = APIRouter(prefix="/chat", tags=["Chat AI"])
logger = get_logger(__name__)


@router.post(
    "",
    response_model=None,
    summary="Tạo cuộc trò chuyện",
    description="Tạo cuộc trò chuyện mới.",
)
async def create_chat(req: CreateChatCommand) -> JSONResponse:
    """
    Endpoint tạo cuộc trò chuyện.

    Args:
        req (ChatCommand): Command chứa thông tin cuộc trò chuyện cần tạo

    Returns:
        JSONResponse
    Raises:
        HTTPException: Khi có lỗi xảy ra trong quá trình xử lý
    """

    logger.info(f"Tạo cuộc trò chuyện: {req.session_id}")

    try:
        result = await Mediator.send(req)
        return result.to_response()
    except Exception as e:
        logger.error(f"Lỗi khi tạo cuộc trò chuyện: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "",
    response_model=None,
    summary="Lấy lịch sử cuộc trò chuyện",
    description="Lấy lịch sử cuộc trò chuyện.",
)
async def get_chat_histories(
    session_id: str = Query(..., description="ID của phiên trò chuyện"),
) -> JSONResponse:
    """
    Endpoint lấy danh sách phiên trò chuyện.

    Args:
        user_id (str): ID của người dùng

    Returns:
        JSONResponse
    Raises:
        HTTPException: Khi có lỗi xảy ra trong quá trình xử lý
    """

    logger.info(f"Lấy lịch sử cuộc trò chuyện: {session_id}")

    try:
        get_chat_histories_query = GetChatHistoriesQuery(session_id=session_id)
        result = await Mediator.send(get_chat_histories_query)
        return result.to_response()
    except Exception as e:
        logger.error(f"Lỗi khi lấy lịch sử cuộc trò chuyện: {e}")
        raise HTTPException(status_code=500, detail=str(e))
