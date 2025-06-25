from fastapi import APIRouter
from .careplan.routes import router as careplan_router
from .analyze.routes import router as analyze_router
from .chatbot.routes import router as chatbot_router

router = APIRouter()
router.include_router(careplan_router, prefix="/careplan", tags=["CarePlan"])
router.include_router(analyze_router, prefix="/analyze", tags=["Analyze"])
router.include_router(chatbot_router, prefix="/chatbot", tags=["Chatbot"])
