import re


def extract_json(text: str) -> str:
    match = re.search(r"\[\s*{.*}\s*\]", text, re.DOTALL)
    if not match:
        raise ValueError("Không tìm thấy JSON array trong phản hồi")
    return match.group(0)
