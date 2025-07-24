from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from app.api.schemas import DocumentJobStatus, DocumentJobType
from app.database.manager import get_collections
from core.result import Result
from utils import get_logger
from shared.messages.job_message import JobResult

router = APIRouter(prefix="/jobs", tags=["Jobs"])
logger = get_logger(__name__)


@router.get(
    "/documents/active",
    response_model=None,
    summary="Lấy tài liệu đang chờ xử lý",
    description="Lấy tài liệu đang chờ xử lý",
)
async def get_documents_job() -> JSONResponse:
    collections = get_collections()

    document_jobs = await collections.document_jobs.find_one(
        {
            "type": DocumentJobType.UPLOAD,
            "status": DocumentJobStatus.PENDING,
        }
    )

    return Result.success(
        document_jobs, JobResult.FETCHED.code, JobResult.FETCHED.message
    ).to_response()


@router.get(
    "/documents",
    response_model=None,
    summary="Lấy danh sách tài liệu",
    description="Lấy danh sách tài liệu",
)
async def get_documents_job(
    search: str = Query("", description="Tên cơ sở tri thức cần tìm kiếm"),
    sort_by: str = Query("created_at", description="Trường cần sắp xếp"),
    sort_order: str = Query(
        "desc", pattern="^(asc|desc)$", description="Thứ tự sắp xếp: asc hoặc desc"
    ),
    status: DocumentJobStatus = Query(DocumentJobStatus.PENDING),
    type: DocumentJobType = Query(DocumentJobType.UPLOAD),
    page: int = Query(1, ge=1, description="Trang hiện tại"),
    limit: int = Query(10, ge=1, le=100, description="Số lượng bản ghi mỗi trang"),
) -> JSONResponse:
    collections = get_collections()

    query = {
        "type": type,
        "status": status,
    }

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

    document_jobs = await cursor.to_list(length=limit)
    total_count = await collections.document_jobs.count_documents(query)

    return Result.success(
        {
            "items": document_jobs,
            "page": page,
            "limit": limit,
            "total": total_count,
        },
        JobResult.FETCHED.code,
        JobResult.FETCHED.message,
    ).to_response()
