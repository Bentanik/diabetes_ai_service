from fastapi import APIRouter
from models.request import CarePlanRequest
from models.response import CarePlanMeasurementOutResponse
from features.careplan.careplan_generator import generate_careplan_measurements

router = APIRouter()


@router.post("/generate", response_model=list[CarePlanMeasurementOutResponse])
async def generate(request: CarePlanRequest):
    return await generate_careplan_measurements(request)
