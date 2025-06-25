from pydantic import BaseModel
from typing import Optional


class CarePlanMeasurementOutResponse(BaseModel):
    recordType: str  # e.g. BloodGlucose
    period: str  # e.g. morning, before_sleep
    subtype: Optional[str]  # e.g. before_meal, after_meal
    reason: str  # Lý do vì sao cần đo


class MeasurementNoteResponse(BaseModel):
    patientId: str
    recordTime: str
    feedback: str
