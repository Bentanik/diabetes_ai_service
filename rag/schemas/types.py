from enum import Enum
from typing import TypedDict, Optional


class LanguageInfo(TypedDict):
    """Thông tin về ngôn ngữ của văn bản"""

    language: str
    vietnamese_ratio: float
    confidence: float


class StructureType(str, Enum):
    TABLE = "table"
    HIERARCHICAL = "hierarchical"
    COMPLEX = "complex"
    SIMPLE = "simple"


class StructureAnalysis(TypedDict):
    """Kết quả phân tích cấu trúc document"""

    has_headings: bool
    has_lists: bool
    has_tables: bool
    paragraph_count: int
    structure_type: StructureType


class HierarchicalStructure(TypedDict):
    """Thông tin về cấu trúc phân cấp trong document"""

    line_number: int
    level: int
    text: str


class ChunkMetadata(TypedDict, total=False):
    from schemas.pdf.text_block import BlockMetadata

    """Metadata cho từng chunk"""

    source: Optional[str]
    block_metadata: Optional[BlockMetadata]
    strategy: str
    token_count: int
    language_info: LanguageInfo


class Chunk(TypedDict):
    """Một chunk văn bản và metadata đi kèm"""

    text: str
    metadata: ChunkMetadata
