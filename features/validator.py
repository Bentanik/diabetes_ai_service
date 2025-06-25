from constants.careplan_schema import RECORD_TYPES, PERIODS, SUBTYPES_BY_RECORD_TYPE
from models.response import CarePlanMeasurementOutResponse


def validate_careplan_output(data: list[dict]) -> list[CarePlanMeasurementOutResponse]:
    validated = []

    for item in data:
        record_type = item.get("recordType")
        period = item.get("period")
        subtype = item.get("subtype")
        reason = item.get("reason", "")

        if record_type not in RECORD_TYPES:
            raise ValueError(f"recordType không hợp lệ: {record_type}")
        if period not in PERIODS:
            raise ValueError(f"period không hợp lệ: {period}")

        allowed_subtypes = SUBTYPES_BY_RECORD_TYPE.get(record_type, [])
        if subtype is not None and subtype not in allowed_subtypes:
            raise ValueError(
                f"subtype không hợp lệ: {subtype} cho recordType {record_type}"
            )

        validated.append(CarePlanMeasurementOutResponse(**item))

    return validated
