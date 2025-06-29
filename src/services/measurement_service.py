"""Dịch vụ phân tích kết quả đo."""

from datetime import datetime

from core.exceptions import ServiceError
from core.llm_client import get_llm
from core.logging_config import get_logger
from models.request import MeasurementNoteRequest
from models.response import MeasurementNoteResponse
from prompts.measurement_note_prompt import MEASUREMENT_NOTE_PROMPT
from config.settings import get_config

logger = get_logger(__name__)


class MeasurementService:
    """Measurement service."""

    def __init__(self):
        self._llm = get_llm()
        logger.info("Initialized MeasurementService")

    async def analyze_measurement(self, request: MeasurementNoteRequest):
        """Analyze a measurement and provide feedback."""
        try:
            logger.info(f"Analyzing measurement for patient: {request.patientId}")

            # Build prompt with request data
            prompt = self._build_prompt(request)
            logger.debug(
                f"Built measurement analysis prompt for patient {request.patientId}"
            )

            # Query LLM for analysis
            feedback_text = await self._llm.generate(prompt)
            logger.debug(f"Received analysis feedback for patient {request.patientId}")

            # Validate feedback length
            max_length = get_config("max_feedback_length") or 250
            if len(feedback_text) > max_length:
                logger.warning(
                    f"Feedback too long for patient {request.patientId}, truncating"
                )
                feedback_text = feedback_text[:max_length]

            # Create response
            response = MeasurementNoteResponse(
                patientId=request.patientId,
                recordTime=datetime.utcnow().isoformat(),
                feedback=feedback_text.strip(),
            )

            logger.info(
                f"Successfully analyzed measurement for patient {request.patientId}"
            )
            return response

        except ServiceError:
            # Re-raise service errors
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error analyzing measurement for patient {request.patientId}: {e}"
            )
            raise ServiceError(f"Unexpected error in measurement analysis: {e}")

    def _build_prompt(self, request: MeasurementNoteRequest) -> str:
        """Build the analysis prompt from request data."""
        return MEASUREMENT_NOTE_PROMPT.format(
            measurementType=request.measurementType,
            value=request.value,
            time=request.time,
            context=request.context or "Không rõ",
            note=request.note or "Không có ghi chú.",
        )


# Global service instance
_service_instance = None


def get_measurement_service():
    """Get the default measurement analysis service instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = MeasurementService()
    return _service_instance
