"""
Text Block Models - Data structures cho text blocks từ PDF

Module chứa các dataclass và type definitions cho việc trích xuất text từ PDF:
- BlockMetadata: Metadata của một text block
- TextBlock: Một text block hoàn chỉnh với content và metadata  
- PageSize: Kích thước trang PDF
- PageData: Data của một trang PDF hoàn chỉnh
"""

from dataclasses import dataclass
from typing import List, Optional

from ..pdf.bbox import BBox
from ..common import LanguageInfo


@dataclass
class BlockMetadata:
    """
    Metadata cho một text block

    Attributes:
        bbox: Bounding box của text block
        block_type: Loại block (paragraph, heading, etc.)
        num_lines: Số dòng trong block
        num_spans: Số span trong block
        is_cleaned: Text đã được làm sạch hay chưa
        page_index: Index của trang (optional, thêm sau)
    """

    bbox: BBox
    block_type: str
    num_lines: int
    num_spans: int
    is_cleaned: bool
    page_index: Optional[int] = None
    language_info: Optional[LanguageInfo] = None


@dataclass
class TextBlock:
    """
    Một text block hoàn chỉnh từ PDF

    Attributes:
        block_id: ID unique của block
        chunk_context: Nội dung text đã được làm sạch và chunk lại
        context: Nội dung text đã được làm sạch
        metadata: Thông tin metadata của block
    """

    block_id: str
    context: str
    metadata: BlockMetadata


@dataclass
class PageSize:
    """
    Kích thước của trang PDF

    Attributes:
        width: Chiều rộng trang (points)
        height: Chiều cao trang (points)
    """

    width: float
    height: float


@dataclass
class PageData:
    """
    Data hoàn chỉnh của một trang PDF

    Attributes:
        page_index: Index của trang (0-based)
        page_size: Kích thước trang
        blocks: Danh sách các text blocks trong trang
    """

    page_index: int
    page_size: PageSize
    blocks: List[TextBlock]
