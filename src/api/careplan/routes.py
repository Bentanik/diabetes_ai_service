"""API routes kế hoạch chăm sóc với tài liệu Swagger được cải thiện."""

from fastapi import APIRouter, HTTPException
from typing import List

from core.exceptions import ServiceError
from core.logging_config import get_logger
from models.request import CarePlanRequest
from models.response import CarePlanMeasurementOutResponse
from services.care_plan_service import get_care_plan_service

logger = get_logger(__name__)
router = APIRouter(tags=["Kế Hoạch Chăm Sóc"])


@router.post(
    "/generate",
    response_model=List[CarePlanMeasurementOutResponse],
    summary="🎯 Tạo Kế Hoạch Chăm Sóc Cá Nhân",
    description="""
    Tạo kế hoạch chăm sóc tiểu đường cá nhân với khuyến nghị đo lường dựa trên dữ liệu bệnh nhân.
    
    Endpoint này phân tích thông tin bệnh nhân bao gồm:
    - Thông tin nhân khẩu học (tuổi, giới tính, BMI)
    - Loại tiểu đường và phương pháp điều trị
    - Tiền sử bệnh và biến chứng
    - Yếu tố lối sống hiện tại
    
    Trả về danh sách tùy chỉnh các lịch đo với thời gian cụ thể và lý do.
    """,
    response_description="Danh sách khuyến nghị đo lường cá nhân với thời gian và lý do",
    responses={
        200: {
            "description": "Tạo kế hoạch chăm sóc thành công",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "recordType": "BloodGlucose",
                            "period": "before_breakfast",
                            "subtype": "fasting",
                            "reason": "Theo dõi đường huyết lúc đói để đánh giá hiệu quả điều trị insulin ban đêm và khả năng kiểm soát glucose tự nhiên của cơ thể.",
                        },
                        {
                            "recordType": "BloodPressure",
                            "period": "morning",
                            "subtype": "sitting",
                            "reason": "Kiểm tra huyết áp buổi sáng để phát hiện sớm biến chứng tim mạch, đặc biệt quan trọng với bệnh nhân tiểu đường type 2.",
                        },
                    ]
                }
            },
        },
        400: {
            "description": "Dữ liệu yêu cầu không hợp lệ hoặc lỗi dịch vụ",
            "content": {
                "application/json": {
                    "example": {
                        "error": "SERVICE_ERROR",
                        "message": "Dữ liệu bệnh nhân không hợp lệ: tuổi phải từ 1 đến 120",
                    }
                }
            },
        },
        500: {
            "description": "Lỗi máy chủ nội bộ",
            "content": {
                "application/json": {
                    "example": {
                        "error": "INTERNAL_ERROR",
                        "message": "Đã xảy ra lỗi không mong muốn khi tạo kế hoạch chăm sóc",
                    }
                }
            },
        },
    },
)
async def generate_care_plan(
    request: CarePlanRequest,
) -> List[CarePlanMeasurementOutResponse]:
    """Tạo kế hoạch chăm sóc cá nhân cho bệnh nhân tiểu đường."""
    try:
        logger.info(
            f"Nhận yêu cầu tạo kế hoạch chăm sóc cho bệnh nhân: {request.patientId}"
        )

        service = get_care_plan_service()
        result = await service.generate_care_plan(request)

        logger.info(
            f"Tạo kế hoạch chăm sóc thành công cho bệnh nhân: {request.patientId}"
        )
        return result

    except ServiceError as e:
        logger.error(
            f"Lỗi dịch vụ khi tạo kế hoạch chăm sóc cho bệnh nhân {request.patientId}: {e.message}"
        )
        raise HTTPException(
            status_code=400,
            detail={"error": "SERVICE_ERROR", "message": e.message},
        )
    except Exception as e:
        logger.error(
            f"Lỗi không mong muốn khi tạo kế hoạch chăm sóc cho bệnh nhân {request.patientId}: {e}"
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": "INTERNAL_ERROR",
                "message": "Đã xảy ra lỗi không mong muốn khi tạo kế hoạch chăm sóc",
            },
        )
