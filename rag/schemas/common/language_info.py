from dataclasses import dataclass


@dataclass
class LanguageInfo:
    """Thông tin về ngôn ngữ của văn bản"""
    language: str
    vietnamese_ratio: float
    confidence: float