"""Dịch vụ kế hoạch chăm sóc."""

import json

from core.exceptions import ServiceError
from core.llm_client import get_llm
from core.logging_config import get_logger
from features.validator import validate_careplan_output
from models.request import CarePlanRequest
from models.response import CarePlanMeasurementOutResponse
from prompts.careplan_prompt import build_prompt
from utils.utils import extract_json

logger = get_logger(__name__)


class CarePlanService:
    """Care plan service."""

    def __init__(self):
        self._llm = get_llm()
        logger.info("Initialized CarePlanService")

    async def generate_care_plan(self, request: CarePlanRequest):
        """Generate a care plan based on patient data."""
        try:
            logger.info(f"Generating care plan for patient: {request.patientId}")

            # Build prompt
            prompt = build_prompt(request)
            logger.debug(f"Built prompt for patient {request.patientId}")

            # Query LLM
            llm_response = await self._llm.generate(prompt)
            logger.debug(f"Received LLM response for patient {request.patientId}")

            # Parse JSON response
            try:
                json_str = extract_json(llm_response)
                raw_data = json.loads(json_str)
                logger.debug(
                    f"Successfully parsed JSON for patient {request.patientId}"
                )
            except Exception as e:
                logger.error(
                    f"JSON parsing failed for patient {request.patientId}: {e}"
                )
                raise ServiceError(f"Failed to parse LLM response as JSON: {e}")

            # Validate and convert to response models
            try:
                validated_data = validate_careplan_output(raw_data)
                logger.info(
                    f"Successfully generated {len(validated_data)} measurements for patient {request.patientId}"
                )
                return validated_data
            except Exception as e:
                logger.error(f"Validation failed for patient {request.patientId}: {e}")
                raise ServiceError(f"Care plan validation failed: {e}")

        except ServiceError:
            # Re-raise service errors
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error generating care plan for patient {request.patientId}: {e}"
            )
            raise ServiceError(f"Unexpected error in care plan generation: {e}")


# Global service instance
_service_instance = None


def get_care_plan_service():
    """Get the default care plan service instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = CarePlanService()
    return _service_instance
