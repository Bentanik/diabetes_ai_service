from fastapi import APIRouter, HTTPException
from models.request import MeasurementNoteRequest
from features.measurement_note.analyze_measurement_service import (
    analyze_measurement_service,
)

router = APIRouter()


@router.post("/analyze-measurement-note")
async def analyze_measurement_note(req: MeasurementNoteRequest):
    try:
        return await analyze_measurement_service(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
