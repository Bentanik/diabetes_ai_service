from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from math import ceil
from app.api.schemas import DocumentJobStatus, DocumentJobType
from app.database.manager import get_collections
from app.database.models.document_job_model import DocumentJobModel
from app.dto.pagination import Pagination
from core.result import Result
from utils import get_logger
from shared.messages.job_message import JobResult

router = APIRouter(prefix="/api/v1/jobs", tags=["Jobs"])
logger = get_logger(__name__)


@router.get(
    "/documents/history",
    summary="Lấy lịch sử xử lý tài liệu",
    description="Lấy lịch sử xử lý tài liệu",
)
async def get_document_history(
    search: str = Query("", description="Tên cơ sở tri thức cần tìm kiếm"),
    type: Optional[DocumentJobType] = Query(None, description="Loại tài liệu"),
    status: Optional[DocumentJobStatus] = Query(None, description="Trạng thái"),
    sort_by: str = Query("created_at", description="Trường cần sắp xếp"),
    sort_order: str = Query(
        "desc", pattern="^(asc|desc)$", description="Thứ tự sắp xếp"
    ),
    page: int = Query(1, ge=1, description="Trang hiện tại"),
    limit: int = Query(10, ge=1, le=100, description="Số lượng bản ghi mỗi trang"),
) -> JSONResponse:
    collections = get_collections()

    query = {}
    if type:
        query["type"] = type
    if status:
        # status là nested object trong document_jobs
        query["status.status"] = status
    if search:
        query["title"] = {"$regex": search, "$options": "i"}

    skip = (page - 1) * limit
    sort_direction = -1 if sort_order == "desc" else 1

    cursor = (
        collections.document_jobs.find(query)
        .sort(sort_by, sort_direction)
        .skip(skip)
        .limit(limit)
    )

    # Lấy dữ liệu từ DB, convert sang model rồi serialize về dict có thể trả về API
    document_jobs_raw = await cursor.to_list(length=limit)
    models = [DocumentJobModel.from_dict(job) for job in document_jobs_raw]

    def serialize_job(model: DocumentJobModel) -> dict:
        return {
            "id": model.id,
            "document_id": model.document_id,
            "knowledge_id": model.knowledge_id,
            "title": model.title,
            "description": model.description,
            "type": getattr(model.type, "value", model.type),
            "status": (
                model.status.to_dict()
                if hasattr(model.status, "to_dict")
                else model.status
            ),
            "file": (
                model.file.to_dict() if hasattr(model.file, "to_dict") else model.file
            ),
            "priority_diabetes": model.priority_diabetes,
            "is_document_delete": model.is_document_delete,
            "is_document_duplicate": model.is_document_duplicate,
            "created_at": model.created_at,
            "updated_at": model.updated_at,
        }

    document_jobs = [serialize_job(m) for m in models]

    total_count = await collections.document_jobs.count_documents(query)
    total_pages = ceil(total_count / limit)

    pagination = Pagination(
        items=document_jobs,
        total=total_count,
        page=page,
        limit=limit,
        total_pages=total_pages,
    )

    return Result.success(
        pagination, JobResult.FETCHED.code, JobResult.FETCHED.message
    ).to_response()
