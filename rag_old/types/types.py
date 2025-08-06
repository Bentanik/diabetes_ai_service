from typing import TypedDict


class LanguageInfo(TypedDict):
    """Thông tin về ngôn ngữ của văn bản"""

    language: str
    vietnamese_ratio: float
    confidence: float


class StructureAnalysis(TypedDict):
    """Kết quả phân tích cấu trúc document"""

    has_headings: bool
    has_lists: bool
    has_tables: bool
    has_code: bool
    paragraph_count: int
    structure_type: str


class HierarchicalStructure(TypedDict):
    """Thông tin về cấu trúc phân cấp trong document"""

    line_number: int
    level: int
    text: str
