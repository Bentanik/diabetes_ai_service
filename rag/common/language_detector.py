import re
from functools import lru_cache

from ..schemas.common import LanguageType, LanguageInfo

class LanguageDetector:
    @staticmethod
    @lru_cache(maxsize=1000)
    def detect_language(text: str) -> LanguageInfo:
        if not text:
            return {"language": LanguageType.UNKNOWN, "vietnamese_ratio": 0.0, "confidence": 0.0}

        vietnamese_chars = (
            r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđĐ]"
        )
        vietnamese_count = len(re.findall(vietnamese_chars, text, re.IGNORECASE))
        total_chars = len(
            re.findall(
                r"[a-zA-ZàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđĐ]",
                text,
            )
        )

        if total_chars == 0:
            return {"language": LanguageType.UNKNOWN, "vietnamese_ratio": 0.0, "confidence": 0.0}

        vietnamese_ratio = vietnamese_count / total_chars

        if vietnamese_ratio > 0.20:
            language = LanguageType.VIETNAMESE
            confidence = min(vietnamese_ratio * 2, 1.0)
        elif vietnamese_ratio > 0.05:
            language = LanguageType.MIXED
            confidence = 0.7
        else:
            language = LanguageType.ENGLISH
            confidence = 1.0 - vietnamese_ratio

        return {
            "language": language,
            "vietnamese_ratio": vietnamese_ratio,
            "confidence": confidence,
        }
