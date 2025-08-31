from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from core.cqrs import Command

@dataclass
class CreateHealthRecordCommand(Command):
    user_id: str
    patient_id: str
    type: str
    value: float
    unit: str
    subtype: Optional[str] = None
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if not self.user_id:
            raise ValueError("user_id không được để trống")
        if not self.patient_id:
            raise ValueError("patient_id không được để trống")
        if not self.type:
            raise ValueError("type không được để trống")
        if self.value < 0:
            raise ValueError("value không được để trống")