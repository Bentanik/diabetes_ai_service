# import urllib
# from bson import ObjectId
# from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
# from fastapi.responses import JSONResponse
# from fastapi.responses import StreamingResponse

# from app.database import get_collections
# from app.feature.document import (
#     CreateDocumentCommand,
# )
# from core.cqrs import Mediator
# from utils import (
#     get_logger,
# )

# router = APIRouter(tags=["RAG"])
# logger = get_logger(__name__)


# @router.post(
#     "/train",
#     response_model=None,
#     summary="Huấn luyện mô hình RAG",
#     description="Huấn luyện mô hình RAG",
# )
# async def train_rag(
#     document_id: str = Query(...),
# ) -> JSONResponse:
#     logger.info(f"Huấn luyện mô hình RAG: {document_id}")
#     try:
#         return
#     except Exception as e:
#         logger.error(f"Lỗi tạo tài liệu: {str(e)}", exc_info=True)
#         raise HTTPException(status_code=500, detail="Tạo tài liệu thất bại")
