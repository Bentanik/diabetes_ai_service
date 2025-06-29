"""DEPRECATED: Legacy measurement analysis service. Use services.measurement_service instead."""

import warnings

from models.request import MeasurementNoteRequest
from models.response import MeasurementNoteResponse
from services.measurement_service import get_measurement_service

warnings.warn(
    "features.measurement_note.analyze_measurement_service is deprecated. Use services.measurement_service instead.",
    DeprecationWarning,
    stacklevel=2,
)


async def analyze_measurement_service(
    req: MeasurementNoteRequest,
) -> MeasurementNoteResponse:
    """Legacy function for measurement analysis.

    DEPRECATED: Use services.measurement_service.get_measurement_service().analyze_measurement() instead.
    """
    warnings.warn(
        "analyze_measurement_service is deprecated. Use MeasurementAnalysisService.analyze_measurement() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    service = get_measurement_service()
    return await service.analyze_measurement(req)
