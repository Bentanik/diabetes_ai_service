from fastapi import APIRouter
from models.request import CarePlanRequest, ChatRequest
from models.response import CarePlanMeasurementOutResponse
from features.careplan.careplan_generator import generate_careplan_measurements

router = APIRouter()


@router.post("/chat")
async def chat_with_agent(req: ChatRequest):
    # result = run_chatbot_agent(req.message)
    return {"reply": req.message}
