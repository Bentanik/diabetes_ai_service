import urllib
from bson import ObjectId
from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse

from app.database import get_collections
from app.feature.document import (
    CreateDocumentCommand,
    GetDocumentQuery,
    GetDocumentsQuery,
)
from app.storage import minio_manager
from core.cqrs import Mediator
from core.result import Result
from shared.messages import DocumentResult
from utils import (
    compress_stream,
    get_best_compression,
    get_logger,
    should_compress,
)

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


@router.get(
    "",
    response_model=None,
    summary="Lấy danh sách tài liệu",
    description="Lấy danh sách tài liệu",
)
async def get_documents(
    search: str = Query("", description="Tên cơ sở tri thức cần tìm kiếm"),
    sort_by: str = Query("created_at", description="Trường cần sắp xếp"),
    sort_order: str = Query(
        "desc", pattern="^(asc|desc)$", description="Thứ tự sắp xếp: asc hoặc desc"
    ),
    page: int = Query(1, ge=1, description="Trang hiện tại"),
    limit: int = Query(10, ge=1, le=100, description="Số lượng bản ghi mỗi trang"),
) -> JSONResponse:
    logger.info(f"Lấy danh sách cơ sở tri thức - search={search}, page={page}")
    try:
        query = GetDocumentsQuery(
            search=search,
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


@router.get(
    "/{document_id}",
    response_model=None,
    summary="Lấy thông tin tài liệu",
    description="Lấy thông tin tài liệu",
)
async def get_document(document_id: str) -> JSONResponse:
    logger.info(f"Lấy thông tin tài liệu: {document_id}")
    try:
        query = GetDocumentQuery(id=document_id)
        result = await Mediator.send(query)
        return result.to_response()
    except Exception as e:
        logger.error(f"Lỗi lấy danh sách cơ sở tri thức: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Không thể lấy danh sách cơ sở tri thức"
        )


@router.get("/{document_id}/download")
async def download_document(
    document_id: str,
    request: Request,
    compress: bool = Query(False),
    compression_type: str = Query("gzip"),
    chunk_size: int = Query(64 * 1024, ge=8192, le=1024 * 1024),
):
    logger.info(f"Tải tài liệu: {document_id}")
    try:
        if not ObjectId.is_valid(document_id):
            return Result.failure(
                message=DocumentResult.NOT_FOUND.message,
                code=DocumentResult.NOT_FOUND.code,
            )

        collections = get_collections()
        document = await collections.documents.find_one({"_id": ObjectId(document_id)})
        if not document:
            return Result.failure(
                message=DocumentResult.NOT_FOUND.message,
                code=DocumentResult.NOT_FOUND.code,
            )

        file_path_parts = document["file_path"].split("/", 1)
        bucket_name = file_path_parts[0]
        object_name = file_path_parts[1] if len(file_path_parts) > 1 else ""
        print(bucket_name, object_name)
        stream_info = await minio_manager.get_stream_async(
            bucket_name, object_name, chunk_size
        )

        filename = document.get("file_name", stream_info["filename"])
        file_size = stream_info["size"]

        compression_method = None
        if compress and should_compress(filename, file_size):
            accept_encoding = request.headers.get("Accept-Encoding", "")
            compression_method = get_best_compression(accept_encoding, compression_type)

        if compression_method:
            processed_stream = compress_stream(
                stream_info["stream"], compression_method
            )
            headers = {
                "Content-Disposition": f"attachment; filename*=UTF-8''{urllib.parse.quote(filename)}",
                "Content-Encoding": compression_method,
                "Transfer-Encoding": "chunked",
                "X-Original-Size": str(file_size),
            }
        else:
            processed_stream = stream_info["stream"]
            headers = {
                "Content-Disposition": f"attachment; filename*=UTF-8''{urllib.parse.quote(filename)}",
                "Content-Length": str(file_size),
            }

        headers.update(
            {
                "Content-Type": "application/octet-stream",
                "Cache-Control": "public, max-age=3600",
            }
        )

        return StreamingResponse(processed_stream, headers=headers)
    except Exception as e:
        logger.error(f"Lỗi tải tài liệu: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Tải tài liệu thất bại")
