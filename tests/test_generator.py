import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from models.careplan_input import CarePlanInput
from services.careplan_generator import generate_careplan_measurements


@pytest.mark.asyncio
async def test_generate_careplan_measurements_success():
    input_data = CarePlanInput(
        patientId="test-id",
        age=45,
        gender="Nam",
        bmi=24.5,
        diabetesType="Tuýp 2",
        insulinSchedule="2 lần/ngày",
        treatmentMethod="Tiêm Insulin",
        complications=["Thận", "Khác"],
        pastDiseases=["Huyết áp cao"],
        lifestyle="Ít vận động",
    )

    result = await generate_careplan_measurements(input_data)

    assert isinstance(result, list)
    assert len(result) > 0

    for item in result:
        assert item.recordType in ["BloodGlucose", "BloodPressure"]
        assert item.period is not None
        assert isinstance(item.reason, str)
