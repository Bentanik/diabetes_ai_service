"""Enhanced validation with improved error handling and logging."""

from typing import List, Dict, Any

from constants.careplan_schema import RECORD_TYPES, PERIODS, SUBTYPES_BY_RECORD_TYPE
from core.exceptions import ServiceError
from core.logging_config import get_logger
from models.response import CarePlanMeasurementOutResponse
from config.settings import get_config

logger = get_logger(__name__)


def validate_careplan_output(
    data: List[Dict[str, Any]]
) -> List[CarePlanMeasurementOutResponse]:
    """Validate care plan output data with comprehensive error checking.

    Args:
        data: Raw care plan data from LLM

    Returns:
        List of validated care plan measurements

    Raises:
        ValidationException: If validation fails
    """
    if not data:
        logger.warning("Empty care plan data received")
        return []

    if not isinstance(data, list):
        logger.error(f"Care plan data is not a list: {type(data)}")
        raise ServiceError("Care plan data must be a list")

    validated = []

    for idx, item in enumerate(data):
        try:
            validated_item = _validate_single_measurement(item, idx)
            validated.append(validated_item)
        except ServiceError as e:
            logger.error(f"Validation failed for item {idx}: {e.message}")
            raise ServiceError(f"Validation failed for item {idx}: {e.message}")

    logger.info(f"Successfully validated {len(validated)} care plan measurements")
    return validated


def _validate_single_measurement(
    item: Dict[str, Any], index: int
) -> CarePlanMeasurementOutResponse:
    """Validate a single measurement item.

    Args:
        item: Single measurement data
        index: Item index for error reporting

    Returns:
        Validated measurement response

    Raises:
        ValidationException: If validation fails
    """
    if not isinstance(item, dict):
        raise ServiceError(f"Item {index} is not a dictionary: {type(item)}")

    # Required fields validation
    record_type = item.get("recordType")
    period = item.get("period")
    subtype = item.get("subtype")
    reason = item.get("reason", "")

    # Validate recordType
    if not record_type:
        raise ServiceError(f"recordType is missing for item {index}")

    if record_type not in RECORD_TYPES:
        raise ServiceError(f"recordType không hợp lệ: {record_type} for item {index}")

    # Validate period
    if not period:
        raise ServiceError(f"period is missing for item {index}")

    if period not in PERIODS:
        raise ServiceError(f"period không hợp lệ: {period} for item {index}")

    # Validate subtype
    allowed_subtypes = SUBTYPES_BY_RECORD_TYPE.get(record_type, [])
    if subtype is not None and subtype not in allowed_subtypes:
        raise ServiceError(
            f"subtype không hợp lệ: {subtype} cho recordType {record_type} for item {index}"
        )

    # Validate reason
    if not reason or not isinstance(reason, str):
        raise ServiceError(f"reason is missing or invalid for item {index}")

    max_length = get_config("max_reason_length") or 150
    if len(reason) > max_length:
        logger.warning(f"Reason too long for item {index}, truncating")
        reason = reason[:max_length]

    # Create and return validated response
    try:
        return CarePlanMeasurementOutResponse(
            recordType=record_type,
            period=period,
            subtype=subtype,
            reason=reason.strip(),
        )
    except Exception as e:
        raise ServiceError(f"Failed to create response model for item {index}: {e}")
