from fastapi import APIRouter
from . import knowledge_routes

router = APIRouter(prefix="/api/v1/rag")

router.include_router(knowledge_routes.router)
__all__ = ["router"]
