from dataclasses import dataclass
from enum import Enum


class StructureType(Enum):
    TABLE = "table"
    HIERARCHICAL = "hierarchical"
    COMPLEX = "complex"
    SIMPLE = "simple"


@dataclass
class StructureAnalysis:
    """Kết quả phân tích cấu trúc document"""
    has_headings: bool
    has_lists: bool
    has_tables: bool
    paragraph_count: int
    structure_type: StructureType
