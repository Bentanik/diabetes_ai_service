from pydantic import BaseModel
from typing import Optional


class CarePlanMeasurementOutResponse(BaseModel):
    """Mô hình phản hồi cho khuyến nghị đo lường trong kế hoạch chăm sóc.

    Thuộc tính:
        recordType: Loại đo lường (ví dụ: "BloodGlucose", "BloodPressure")
        period: Thời điểm đo (ví dụ: "before_breakfast", "morning")
        subtype: Bối cảnh đo lường cụ thể (ví dụ: "fasting", "sitting")
        reason: Giải thích chi tiết bằng tiếng Việt về lý do khuyến nghị đo lường này
    """

    recordType: str
    period: str
    subtype: Optional[str]
    reason: str

    class Config:
        schema_extra = {
            "example": {
                "recordType": "BloodGlucose",
                "period": "before_breakfast",
                "subtype": "fasting",
                "reason": "Theo dõi đường huyết lúc đói để đánh giá hiệu quả điều trị insulin ban đêm và khả năng kiểm soát glucose tự nhiên của cơ thể.",
            }
        }


class MeasurementNoteResponse(BaseModel):
    """Mô hình phản hồi cho phản hồi phân tích kết quả đo.

    Thuộc tính:
        patientId: Mã định danh bệnh nhân
        recordTime: Dấu thời gian ISO khi thực hiện phân tích
        feedback: Phản hồi cá nhân hóa được tạo bởi AI bằng tiếng Việt
    """

    patientId: str
    recordTime: str
    feedback: str

    class Config:
        schema_extra = {
            "example": {
                "patientId": "P001",
                "recordTime": "2024-01-15T08:30:00.000Z",
                "feedback": "Chỉ số đường huyết 7.2 mmol/L lúc đói của bạn hơi cao hơn mức bình thường (< 7.0). Có thể do bạn ăn tối muộn và căng thẳng công việc. Hãy thử ăn tối sớm hơn, tránh thức khuya và tập thể dục nhẹ buổi tối. Bạn đang cố gắng rất tốt, cứ tiếp tục theo dõi nhé!",
            }
        }


class HealthResponse(BaseModel):
    """Mô hình phản hồi cho endpoint kiểm tra sức khỏe.

    Thuộc tính:
        status: Trạng thái sức khỏe dịch vụ ("healthy" hoặc "unhealthy")
        service: Tên và mô tả dịch vụ
        version: Phiên bản hiện tại của dịch vụ
    """

    status: str
    service: str
    version: str

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "service": "AI Service - CarePlan Generator",
                "version": "1.0.0",
            }
        }
