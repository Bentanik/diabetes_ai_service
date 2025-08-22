from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from app.feature.sessions import (
    CreateSessionCommand,
    UpdateSessionCommand,
    DeleteSessionCommand,
    GetSessionsQuery,
)
from core.cqrs import Mediator
from utils import (
    get_logger,
)

# Khởi tạo router với prefix và tag
router = APIRouter(prefix="/session-chat", tags=["Session Chat AI"])
logger = get_logger(__name__)


@router.post(
    "",
    response_model=None,
    summary="Tạo phiên trò chuyện",
    description="Tạo phiên trò chuyện mới.",
)
async def create_session(req: CreateSessionCommand) -> JSONResponse:
    """
    Endpoint thêm tài liệu vào vector database để huấn luyện AI.

    Args:
        req (AddTrainingDocumentCommand): Command chứa thông tin tài liệu cần thêm

    Returns:
        JSONResponse
    Raises:
        HTTPException: Khi có lỗi xảy ra trong quá trình xử lý
    """

    logger.info(f"Tạo phiên trò chuyện: {req.title}")

    try:
        result = await Mediator.send(req)
        return result.to_response()
    except Exception as e:
        logger.error(f"Lỗi khi tạo phiên trò chuyện: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put(
    "",
    response_model=None,
    summary="Cập nhật phiên trò chuyện",
    description="Cập nhật phiên trò chuyện.",
)
async def update_session(req: UpdateSessionCommand) -> JSONResponse:
    """
    Endpoint cập nhật phiên trò chuyện.

    Args:
        req (UpdateSessionCommand): Command chứa thông tin phiên trò chuyện cần cập nhật

    Returns:
        JSONResponse
    Raises:
        HTTPException: Khi có lỗi xảy ra trong quá trình xử lý
    """

    logger.info(f"Cập nhật phiên trò chuyện: {req.title}")

    try:
        result = await Mediator.send(req)
        return result.to_response()
    except Exception as e:
        logger.error(f"Lỗi khi cập nhật phiên trò chuyện: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "",
    response_model=None,
    summary="Xóa phiên trò chuyện",
    description="Xóa phiên trò chuyện.",
)
async def delete_session(
    session_id: str = Query(..., description="ID của phiên trò chuyện"),
) -> JSONResponse:
    """
    Endpoint xóa phiên trò chuyện.

    Args:
        req (DeleteSessionCommand): Command chứa thông tin phiên trò chuyện cần xóa

    Returns:
        JSONResponse
    Raises:
        HTTPException: Khi có lỗi xảy ra trong quá trình xử lý
    """

    logger.info(f"Xóa phiên trò chuyện: {session_id}")

    try:
        delete_session_command = DeleteSessionCommand(session_id=session_id)
        result = await Mediator.send(delete_session_command)
        return result.to_response()
    except Exception as e:
        logger.error(f"Lỗi khi cập nhật phiên trò chuyện: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "",
    response_model=None,
    summary="Lấy danh sách phiên trò chuyện",
    description="Lấy danh sách phiên trò chuyện.",
)
async def get_sessions(
    user_id: str = Query(..., description="ID của người dùng"),
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

    logger.info(f"Lấy danh sách phiên trò chuyện: {user_id}")

    try:
        get_sessions_query = GetSessionsQuery(user_id=user_id)
        result = await Mediator.send(get_sessions_query)
        return result.to_response()
    except Exception as e:
        logger.error(f"Lỗi khi lấy danh sách phiên trò chuyện: {e}")
        raise HTTPException(status_code=500, detail=str(e))
