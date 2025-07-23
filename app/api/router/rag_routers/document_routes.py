from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.feature.document import CreateDocumentCommand
from utils import get_logger
from core.cqrs import Mediator

router = APIRouter(prefix="/documents", tags=["Documents"])
logger = get_logger(__name__)


@router.post(
    "",
    response_model=None,
    summary="Tạo tài liệu mới",
    description="Tạo mới tài liệu trong hệ thống.",
)
async def create_document(
    file: UploadFile = File(...),
    knowledge_id: str = Form(...),
    title: str = Form(...),
    description: str = Form(...),
) -> JSONResponse:
    logger.info(f"Tạo tài liệu mới: {title}")
    try:
        doc_req = CreateDocumentCommand(
            file=file, knowledge_id=knowledge_id, title=title, description=description
        )
        result = await Mediator.send(doc_req)
        return result.to_response()
    except Exception as e:
        logger.error(f"Lỗi tạo tài liệu: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Tạo tài liệu thất bại")
