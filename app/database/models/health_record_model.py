from datetime import datetime
from typing import Dict, Any, Optional
from app.database.models import BaseModel


class HealthRecordModel(BaseModel):
    def __init__(
        self,
        user_id: str,
        patient_id: str,
        type: str,            # "BloodGlucose" hoặc "BloodPressure"
        value: float,
        unit: str,            # "mmol/l" hoặc "mmHg"
        subtype: Optional[str] = None, # "tâm thu" hoặc "tâm trương"
        timestamp: Optional[datetime] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.user_id = user_id
        self.patient_id = patient_id
        self.type = type
        self.value = value
        self.unit = unit
        self.subtype = subtype
        self.timestamp = timestamp

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Optional["HealthRecordModel"]:
        if data is None:
            return None

        data = dict(data)

        return cls(
            user_id=str(data.pop("user_id", "")),
            patient_id=str(data.pop("patient_id", "")),
            type=data.pop("type", ""),
            value=float(data.pop("value", 0.0)),
            unit=data.pop("unit", ""),
            subtype=data.pop("subtype", None),
            timestamp=data.pop("timestamp", None),
            **data
        )
