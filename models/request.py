from pydantic import BaseModel
from typing import List, Optional


class CarePlanRequest(BaseModel):
    patientId: str
    age: int
    gender: str
    bmi: float
    diabetesType: str
    insulinSchedule: str
    treatmentMethod: str
    complications: List[str]
    pastDiseases: List[str]
    lifestyle: str


class MeasurementNoteRequest(BaseModel):
    patientId: str
    measurementType: str  # e.g., "Blood Glucose", "Blood Pressure"
    value: str  # e.g., "7.8 mmol/L", "145/90 mmHg"
    time: str  # e.g., "07:00", "21:30" (24h format)
    context: Optional[str] = None  # e.g., "fasting", "after lunch", "resting"
    note: Optional[str] = None  # patient's note like diet, sleep, etc.


class ChatRequest(BaseModel):
    message: str
    session_id: str
    patient_id: str
