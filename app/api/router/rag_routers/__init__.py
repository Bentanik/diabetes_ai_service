from fastapi import APIRouter
from . import knowledge_routes, document_routes, train_ai_routes, session_chat_routes, chat_routes, setting_routes

router = APIRouter(prefix="/api/v1/rag")

router.include_router(knowledge_routes.router)
router.include_router(document_routes.router)
router.include_router(train_ai_routes.router)
router.include_router(session_chat_routes.router)
router.include_router(chat_routes.router)
router.include_router(setting_routes.router)

__all__ = ["router"]
