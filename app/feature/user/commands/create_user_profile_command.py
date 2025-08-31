from dataclasses import dataclass, field
from typing import List, Optional
from core.cqrs import Command

@dataclass
class CreateUserProfileCommand(Command):
    """
    Command để thêm hồ sơ người dùng vào database.

    Attributes:
        user_id (str): ID của người dùng
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

    user_id: str
    patient_id: str
    full_name: str
    age: int
    gender: str
    bmi: float
    diabetes_type: str
    insulin_schedule: str
    treatment_method: str
    complications: Optional[List[str]] = field(default_factory=list)
    past_diseases: Optional[List[str]] = field(default_factory=list)
    lifestyle: str = ""

    def __post_init__(self):
        if not self.user_id:
            raise ValueError("user_id không được để trống")
        if not self.patient_id:
            raise ValueError("patient_id không được để trống")
        if self.age < 0:
            raise ValueError("Tuổi phải lớn hơn hoặc bằng 0")
        if self.bmi < 0:
            raise ValueError("BMI phải lớn hơn hoặc bằng 0")
