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
from shared.messages import DocumentJobMessage

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
    knowledge_id: Optional[str] = Query(None, description="ID cơ sở tri thức"),
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
    if knowledge_id:
        query["knowledge_id"] = knowledge_id

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

    document_jobs = [m.to_dict() for m in models]

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
        pagination, DocumentJobMessage.FETCHED.code, DocumentJobMessage.FETCHED.message
    ).to_response()
