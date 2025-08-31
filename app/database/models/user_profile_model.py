from typing import List, Dict, Any, Optional
from app.database.models import BaseModel

class UserProfileModel(BaseModel):
    """
    Model cho hồ sơ người dùng (User Profile)

    Attributes:
        patient_id (str): ID bệnh nhân
        age (int): Tuổi
        gender (str): Giới tính
        bmi (float): BMI
        diabetes_type (str): Loại đái tháo đường
        insulin_schedule (str): Lịch sử dùng insulin
        treatment_method (str): Phương pháp điều trị
        complications (List[str]): Các biến chứng
        past_diseases (List[str]): Các bệnh đã mắc trước đây
        lifestyle (str): Lối sống
    """

    def __init__(
        self,
        user_id: str,
        patient_id: str,
        full_name: str,
        age: int,
        gender: str,
        bmi: float,
        diabetes_type: str,
        insulin_schedule: str,
        treatment_method: str,
        complications: Optional[List[str]] = None,
        past_diseases: Optional[List[str]] = None,
        lifestyle: str = "",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.user_id = user_id
        self.patient_id = patient_id
        self.full_name = full_name
        self.age = age
        self.gender = gender
        self.bmi = bmi
        self.diabetes_type = diabetes_type
        self.insulin_schedule = insulin_schedule
        self.treatment_method = treatment_method
        self.complications = complications or []
        self.past_diseases = past_diseases or []
        self.lifestyle = lifestyle

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserProfileModel":
        if data is None:
            return None

        data = dict(data)

        user_id = str(data.pop("user_id", ""))
        patient_id = str(data.pop("patient_id", ""))
        full_name = str(data.pop("full_name", ""))
        age = int(data.pop("age", 0))
        gender = data.pop("gender", "")
        bmi = float(data.pop("bmi", 0.0))
        diabetes_type = data.pop("diabetes_type", "")
        insulin_schedule = data.pop("insulin_schedule", "")
        treatment_method = data.pop("treatment_method", "")
        complications = data.pop("complications", [])
        past_diseases = data.pop("past_diseases", [])
        lifestyle = data.pop("lifestyle", "")

        return cls(
            user_id=user_id,
            patient_id=patient_id,
            full_name=full_name,
            age=age,
            gender=gender,
            bmi=bmi,
            diabetes_type=diabetes_type,
            insulin_schedule=insulin_schedule,
            treatment_method=treatment_method,
            complications=complications,
            past_diseases=past_diseases,
            lifestyle=lifestyle,
            **data
        )
