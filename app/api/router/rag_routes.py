from fastapi import APIRouter
from fastapi.responses import JSONResponse

from utils import get_logger
from core.cqrs import Mediator
from app.features.knowledge import CreateKnowledgeCommand

router = APIRouter(prefix="/api/v1", tags=["knowledge"])
logger = get_logger(__name__)


@router.post("/knowledge")
async def create_knowledge(
    kb_req: CreateKnowledgeCommand,
) -> JSONResponse:
    """Tạo cơ sở tri thức mới"""

    logger.info(f"Creating knowledge base: {kb_req.name}")

    try:
        result = await Mediator.send(kb_req)

        return result.to_response()

    except Exception as e:
        logger.error(f"Router error: {str(e)}", exc_info=True)
