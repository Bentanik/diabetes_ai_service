"""DEPRECATED: Legacy care plan generator. Use services.care_plan_service instead."""

import warnings
from typing import List

from models.request import CarePlanRequest
from models.response import CarePlanMeasurementOutResponse
from services.care_plan_service import get_care_plan_service

warnings.warn(
    "features.careplan.careplan_generator is deprecated. Use services.care_plan_service instead.",
    DeprecationWarning,
    stacklevel=2,
)


async def generate_careplan_measurements(
    request: CarePlanRequest,
) -> List[CarePlanMeasurementOutResponse]:
    """Legacy function for care plan generation.

    DEPRECATED: Use services.care_plan_service.get_care_plan_service().generate_care_plan() instead.
    """
    warnings.warn(
        "generate_careplan_measurements is deprecated. Use CarePlanService.generate_care_plan() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    service = get_care_plan_service()
    return await service.generate_care_plan(request)
