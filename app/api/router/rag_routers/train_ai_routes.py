from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from app.feature.train_ai import AddTrainingDocumentCommand
from core.cqrs import Mediator
from utils import (
    get_logger,
)

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


# @router.get(
#     "/get-retrieved-context",
#     response_model=None,
#     summary="Lấy context từ vector database",
#     description="Lấy context từ vector database.",
# )
# async def get_retrieved_context(
#     search: str = Query(..., description="Search query string")
# ) -> JSONResponse:
#     """
#     Endpoint lấy context từ vector database.

#     Args:
#         req (GetRetrievedContextQuery): Query chứa query cần lấy context

#     Returns:
#         JSONResponse
#     """
#     try:
#         query = GetRetrievedContextQuery(query=search)
#         result = await Mediator.send(query)
#         return result.to_response()
#     except Exception as e:
#         logger.error(f"Lỗi khi thêm tài liệu vào vector database: {e}")
#         raise HTTPException(status_code=500, detail=str(e))
