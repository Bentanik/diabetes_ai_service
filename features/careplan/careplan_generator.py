import json
from models.request import CarePlanRequest
from models.response import CarePlanMeasurementOutResponse
from core.llm_client import query_care_plan_llm
from features.validator import validate_careplan_output
from prompts.careplan_prompt import build_prompt
from utils.utils import extract_json


async def generate_careplan_measurements(
    request: CarePlanRequest,
) -> list[CarePlanMeasurementOutResponse]:
    prompt = build_prompt(request)

    llm_response = await query_care_plan_llm(prompt)

    try:
        json_str = extract_json(str(llm_response.content))
        raw = json.loads(json_str)
    except Exception as e:
        raise ValueError(f"Lỗi khi parse JSON: {e}")

    validated = validate_careplan_output(raw)
    return validated
