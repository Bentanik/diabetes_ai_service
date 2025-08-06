from dataclasses import dataclass

@dataclass
class HierarchicalStructure:
    """Thông tin về cấu trúc phân cấp trong document"""
    line_number: int
    level: int
    text: str
