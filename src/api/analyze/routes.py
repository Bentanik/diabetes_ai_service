"""API routes phân tích kết quả đo với tài liệu Swagger được cải thiện."""

from fastapi import APIRouter, HTTPException

from core.exceptions import ServiceError
from core.logging_config import get_logger
from models.request import MeasurementNoteRequest
from models.response import MeasurementNoteResponse
from services.measurement_service import get_measurement_service

logger = get_logger(__name__)
router = APIRouter(tags=["Phân Tích Kết Quả Đo"])


@router.post(
    "/analyze-measurement-note",
    response_model=MeasurementNoteResponse,
    summary="📊 Phân Tích Kết Quả Đo Bệnh Nhân",
    description="""
    Phân tích kết quả đo sức khỏe của bệnh nhân và cung cấp phản hồi cá nhân hóa được hỗ trợ bởi AI.
    
    Endpoint này xử lý dữ liệu đo lường bao gồm:
    - Loại đo lường (Đường huyết, Huyết áp, v.v.)
    - Giá trị đo và đơn vị
    - Thời gian đo (định dạng 24h)
    - Bối cảnh (lúc đói, sau ăn, nghỉ ngơi, v.v.)
    - Ghi chú của bệnh nhân (ăn uống, ngủ, căng thẳng, hoạt động)
    
    Trả về phân tích thông minh với:
    - Đánh giá giá trị đo (bình thường/cao/thấp)
    - Giải thích theo bối cảnh dựa trên thời gian và tình huống
    - Khuyến nghị và động viên cá nhân hóa
    - Phản hồi bằng tiếng Việt
    """,
    response_description="Phân tích chi tiết với phản hồi cá nhân hóa bằng tiếng Việt",
    responses={
        200: {
            "description": "Phân tích kết quả đo thành công",
            "content": {
                "application/json": {
                    "example": {
                        "patientId": "P001",
                        "recordTime": "2024-01-15T08:30:00.000Z",
                        "feedback": "Chỉ số đường huyết 7.2 mmol/L lúc đói của bạn hơi cao hơn mức bình thường (< 7.0). Có thể do bạn ăn tối muộn và căng thẳng công việc. Hãy thử ăn tối sớm hơn, tránh thức khuya và tập thể dục nhẹ buổi tối. Bạn đang cố gắng rất tốt, cứ tiếp tục theo dõi nhé!",
                    }
                }
            },
        },
        400: {
            "description": "Dữ liệu đo lường không hợp lệ hoặc lỗi dịch vụ",
            "content": {
                "application/json": {
                    "example": {
                        "error": "SERVICE_ERROR",
                        "message": "Định dạng giá trị đo không hợp lệ",
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
                        "message": "Đã xảy ra lỗi không mong muốn khi phân tích kết quả đo",
                    }
                }
            },
        },
    },
)
async def analyze_measurement_note(
    request: MeasurementNoteRequest,
) -> MeasurementNoteResponse:
    """Phân tích kết quả đo sức khỏe của bệnh nhân với phản hồi được hỗ trợ bởi AI."""
    try:
        logger.info(
            f"Nhận yêu cầu phân tích kết quả đo cho bệnh nhân: {request.patientId}"
        )

        service = get_measurement_service()
        result = await service.analyze_measurement(request)

        logger.info(
            f"Phân tích kết quả đo thành công cho bệnh nhân: {request.patientId}"
        )
        return result

    except ServiceError as e:
        logger.error(
            f"Lỗi dịch vụ khi phân tích kết quả đo cho bệnh nhân {request.patientId}: {e.message}"
        )
        raise HTTPException(
            status_code=400,
            detail={"error": "SERVICE_ERROR", "message": e.message},
        )
    except Exception as e:
        logger.error(
            f"Lỗi không mong muốn khi phân tích kết quả đo cho bệnh nhân {request.patientId}: {e}"
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": "INTERNAL_ERROR",
                "message": "Đã xảy ra lỗi không mong muốn khi phân tích kết quả đo",
            },
        )
