"""
Language Info - Value Object cho thông tin ngôn ngữ

File này định nghĩa LanguageInfo để lưu trữ và xử lý thông tin về
ngôn ngữ của tài liệu.
"""

from dataclasses import dataclass
from typing import Dict, Any

from app.database.enums import LanguageType


@dataclass
class LanguageInfo:
    """
    Value Object chứa thông tin về ngôn ngữ

    Attributes:
        language (str): Ngôn ngữ của tài liệu
        confidence (float): Độ tin cậy của việc xác định ngôn ngữ
    """

    language: LanguageType
    vietnamese_ratio: float
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi sang dictionary cho MongoDB"""
        return {
            "language": self.language,
            "vietnamese_ratio": self.vietnamese_ratio,
            "confidence": self.confidence,
        }
