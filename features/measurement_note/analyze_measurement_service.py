from datetime import datetime
from models.request import MeasurementNoteRequest
from core.llm_client import query_note_record_llm
from prompts.measurement_note_prompt import MEASUREMENT_NOTE_PROMPT


async def analyze_measurement_service(req: MeasurementNoteRequest):
    # Fill prompt
    prompt = MEASUREMENT_NOTE_PROMPT.format(
        measurementType=req.measurementType,
        value=req.value,
        time=req.time,
        context=req.context or "Không rõ",
        note=req.note or "Không có ghi chú.",
    )

    # Query LLM
    res = await query_note_record_llm(prompt)

    # Extract text safely
    text = (
        res.content.strip()
        if hasattr(res, "content") and isinstance(res.content, str)
        else (
            res.strip()
            if isinstance(res, str)
            else (
                res[0].strip()
                if isinstance(res, list) and res and isinstance(res[0], str)
                else None
            )
        )
    )
    if not text:
        raise Exception("⚠️ LLM did not return a valid response")

    return {
        "patientId": req.patientId,
        "recordTime": datetime.utcnow().isoformat(),
        "feedback": text,
    }
