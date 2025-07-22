from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    Path,
    Request,
    UploadFile,
    HTTPException,
    Query,
    status,
    Depends,
)

from app.api.schemas import (
    SuccessResponse,
    ErrorResponse,
    KnowledgeBaseCreateRequest,
)

from app.database.models.knowledge_model import KnowledgeModel
from utils import get_logger

from app.database import get_collections


logger = get_logger(__name__)

router = APIRouter(prefix="/rag", tags=["RAG"])


@router.post(
    "/knowledge-bases",
    response_model=SuccessResponse | ErrorResponse,
    summary="Tạo cơ sở tri thức mới",
    description="Tạo mới một cơ sở tri thức. Mỗi cơ sở tri thức đại diện cho một lĩnh vực nghiên cứu riêng biệt.",
)
async def create_knowledge_base(kb_req: KnowledgeBaseCreateRequest):
    try:
        collections = get_collections()

        new_kb = KnowledgeModel(name=kb_req.name, description=kb_req.description)

        print(new_kb._id)
        await collections.knowledges.insert_one(new_kb.to_dict())
        return SuccessResponse(
            isSuccess=True,
            code="SUCCESS",
            message="Tạo cơ sở tri thức thành công",
            data=None,
        )
    except Exception as e:
        logger.error(f"Lỗi không tạo được cơ sở tri thức: {e}")
        raise HTTPException(500, detail=f"Lỗi không tạo được cơ sở tri thức: {str(e)}")
