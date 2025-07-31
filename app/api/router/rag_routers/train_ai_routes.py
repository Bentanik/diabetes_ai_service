from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from app.feature.train_ai.commands import (
    AddTrainingDocumentCommand,
)
from core.cqrs import Mediator
from utils import (
    get_logger,
)

# Khởi tạo router với prefix và tag
router = APIRouter(prefix="/train-ai", tags=["Train AI"])
logger = get_logger(__name__)


@router.post(
    "",
    response_model=None,
    summary="Thêm tài liệu vào vector database",
    description="Thêm tài liệu vào vector database để huấn luyện AI.",
)
async def add_training_document(req: AddTrainingDocumentCommand) -> JSONResponse:
    """
    Endpoint thêm tài liệu vào vector database để huấn luyện AI.

    Args:
        req (AddTrainingDocumentCommand): Command chứa thông tin tài liệu cần thêm

    Returns:
        JSONResponse
    Raises:
        HTTPException: Khi có lỗi xảy ra trong quá trình xử lý
    """

    logger.info(f"Thêm tài liệu vào vector database: {req.document_id}")

    try:
        result = await Mediator.send(req)
        return result.to_response()
    except Exception as e:
        logger.error(f"Lỗi khi thêm tài liệu vào vector database: {e}")
        raise HTTPException(status_code=500, detail=str(e))
