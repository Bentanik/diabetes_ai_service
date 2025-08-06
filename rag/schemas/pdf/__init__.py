"""
PDF Types Package - Export all PDF-related data types

Package này chứa tất cả các data structures và types cho việc xử lý PDF:
- BBox: Bounding box cho text regions
- TextBlock, BlockMetadata: Text block và metadata
- PageData, PageSize: Page data và dimensions
"""

from .bbox import BBox
from .text_block import TextBlock, BlockMetadata, PageData, PageSize
from .structure import StructureAnalysis, StructureType
from .hierarchical_structure import HierarchicalStructure

__all__ = [
    "BBox",
    "TextBlock",
    "BlockMetadata",
    "PageData",
    "PageSize",
    "StructureAnalysis",
    "HierarchicalStructure",
    "StructureType",
]
