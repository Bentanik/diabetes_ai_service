from pydantic import BaseModel
from typing import List, Optional


class CarePlanRequest(BaseModel):
    """Mô hình yêu cầu để tạo kế hoạch chăm sóc cá nhân.

    Thuộc tính:
        patientId: Mã định danh bệnh nhân duy nhất
        age: Tuổi bệnh nhân tính bằng năm (1-120)
        gender: Giới tính bệnh nhân
        bmi: Chỉ số khối cơ thể (10.0-50.0)
        diabetesType: Loại tiểu đường
        insulinSchedule: Lịch trình insulin hiện tại
        treatmentMethod: Phương pháp điều trị chính
        complications: Danh sách biến chứng tiểu đường
        pastDiseases: Danh sách bệnh lý đã mắc trước đây
        lifestyle: Mô tả lối sống của bệnh nhân
    """

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

    class Config:
        schema_extra = {
            "example": {
                "patientId": "P001",
                "age": 45,
                "gender": "Nam",
                "bmi": 23.5,
                "diabetesType": "Type 2",
                "insulinSchedule": "Twice daily",
                "treatmentMethod": "Insulin + Metformin",
                "complications": ["Hypertension", "Retinopathy"],
                "pastDiseases": ["Heart disease"],
                "lifestyle": "Sedentary work, irregular meals, high stress",
            }
        }


class MeasurementNoteRequest(BaseModel):
    """Mô hình yêu cầu để phân tích kết quả đo của bệnh nhân.

    Thuộc tính:
        patientId: Mã định danh bệnh nhân duy nhất
        measurementType: Loại đo lường (ví dụ: "Blood Glucose", "Blood Pressure")
        value: Giá trị đo với đơn vị (ví dụ: "7.2 mmol/L", "145/90 mmHg")
        time: Thời gian đo theo định dạng 24h (ví dụ: "07:30", "21:30")
        context: Bối cảnh đo lường (ví dụ: "fasting", "after lunch", "resting")
        note: Ghi chú của bệnh nhân về các điều kiện ảnh hưởng đến kết quả đo
    """

    patientId: str
    measurementType: str
    value: str
    time: str
    context: Optional[str] = None
    note: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "patientId": "P001",
                "measurementType": "Blood Glucose",
                "value": "7.2 mmol/L",
                "time": "07:30",
                "context": "fasting",
                "note": "Ăn tối muộn, ngủ không đủ giấc, căng thẳng công việc",
            }
        }


class ChatRequest(BaseModel):
    """Mô hình yêu cầu cho tương tác chatbot.

    Thuộc tính:
        message: Tin nhắn của người dùng gửi đến chatbot
        session_id: Mã định danh phiên trò chuyện
        patient_id: Mã định danh bệnh nhân
    """

    message: str
    session_id: str
    patient_id: str

    class Config:
        schema_extra = {
            "example": {
                "message": "Tôi cần tư vấn về chế độ ăn cho tiểu đường",
                "session_id": "sess_001",
                "patient_id": "P001",
            }
        }
