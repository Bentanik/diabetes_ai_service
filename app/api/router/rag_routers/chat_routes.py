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
from typing import Optional

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
    session_id: Optional[str] = Query(None, description="ID của phiên trò chuyện"),
    user_id: Optional[str] = Query(None, description="ID của người dùng")
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
        get_chat_histories_query = GetChatHistoriesQuery(session_id=session_id, user_id=user_id)
        result = await Mediator.send(get_chat_histories_query)
        return result.to_response()
    except Exception as e:
        logger.error(f"Lỗi khi lấy lịch sử cuộc trò chuyện: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@router.delete("/delete-chat-admin", 
    response_model=None,
    summary="Xóa cuộc trò chuyện của Admin",
    description="Xóa cuộc trò chuyện của Admin.",
)
async def delete_chat_admin(user_id: str = Query(..., description="ID của người dùng")) -> JSONResponse:
    """
    Endpoint xóa cuộc trò chuyện của Admin.

    Args:
        user_id (str): ID của người dùng

    Returns:
        JSONResponse
    Raises:
        HTTPException: Khi có lỗi xảy ra trong quá trình xử lý
    """

    logger.info(f"Xóa cuộc trò chuyện của Admin: {user_id}")
    from app.database.manager import get_collections
    
    try:
        collection = get_collections()
        result = await collection.chat_histories.delete_many({"user_id": user_id})
        return JSONResponse(
            content={"message": f"Đã xóa {result.deleted_count} cuộc trò chuyện của Admin."},
            status_code=200
        )
    except Exception as e:
        logger.error(f"Lỗi khi xóa cuộc trò chuyện của Admin: {e}")
        raise HTTPException(status_code=500, detail=str(e))