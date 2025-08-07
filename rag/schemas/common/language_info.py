from dataclasses import dataclass
from enum import Enum

class LanguageType(str, Enum):
    UNKNOWN = "unknown"
    MIXED = "mixed"
    VIETNAMESE = "vi"
    ENGLISH = "en"

@dataclass
class LanguageInfo:
    """Thông tin về ngôn ngữ của văn bản"""
    language: LanguageType
    vietnamese_ratio: float
    confidence: float