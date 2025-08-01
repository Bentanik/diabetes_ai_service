from fastapi import APIRouter
from . import knowledge_routes
from . import document_routes
from . import train_ai_routes
from . import session_chat_routes
from . import chat_routes

router = APIRouter(prefix="/api/v1/rag")

router.include_router(knowledge_routes.router)
router.include_router(document_routes.router)
router.include_router(train_ai_routes.router)
router.include_router(session_chat_routes.router)
router.include_router(chat_routes.router)

__all__ = ["router"]
