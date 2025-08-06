import asyncio
import re
import logging
from typing import List
from ..schemas.pdf import StructureAnalysis, HierarchicalStructure, StructureType

logger = logging.getLogger(__name__)


class DocumentStructureAnalyzer:
    """Phân tích cấu trúc document"""

    # Patterns để detect headings
    ALL_HEADING_PATTERNS = [
        r"^(?:CHƯƠNG|PHẦN|MỤC|BÀI|TIẾT|PHỤLỤC|PHỤ LỤC|CHAPTER|SECTION|PART|ARTICLE|APPENDIX)\s+[IVXLCDM\d]+",
        r"^(?:\d+|[IVXLCDM]+)\.\s+[A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐĐ]",
        r"^#{1,6}\s+",
        r"^[A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐĐ][A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐĐ\s]{2,}$",
    ]

    # Patterns để detect lists
    LIST_PATTERNS = [
        r"^(?:[-•*]|\d+[\.\)]|[a-zA-Z][\.\)]|[IVXLCDM]+[\.\)])\s+",
    ]

    async def analyze_structure(self, text: str) -> StructureAnalysis:
        """
        Phân tích cấu trúc của document

        Args:
            text: Nội dung document

        Returns:
            StructureAnalysis với thông tin về cấu trúc
        """
        if not text:
            return {
                "has_headings": False,
                "has_lists": False,
                "has_tables": False,
                "paragraph_count": 0,
                "structure_type": "simple",
            }

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._analyze_structure_sync, text)

    def _analyze_structure_sync(self, text: str) -> StructureAnalysis:
        lines = text.split("\n")
        analysis: StructureAnalysis = {
            "has_headings": False,
            "has_lists": False,
            "has_tables": False,
            "paragraph_count": 0,
            "structure_type": StructureType.SIMPLE,
        }

        heading_count = 0
        list_count = 0
        paragraph_count = 0

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue

            # Check headings
            for pattern in self.ALL_HEADING_PATTERNS:
                if re.match(pattern, line_stripped, re.IGNORECASE):
                    analysis["has_headings"] = True
                    heading_count += 1
                    break

            # Check lists
            for pattern in self.LIST_PATTERNS:
                if re.match(pattern, line_stripped):
                    analysis["has_lists"] = True
                    list_count += 1
                    break

            # Check tables
            if "|" in line and line.count("|") >= 2:
                analysis["has_tables"] = True

            # Count paragraphs
            if len(line_stripped) > 10:
                paragraph_count += 1

        analysis["paragraph_count"] = paragraph_count

        # Determine structure type
        if heading_count > paragraph_count * 0.2:
            analysis["structure_type"] = StructureType.HIERARCHICAL
        elif analysis["has_tables"]:
            analysis["structure_type"] = StructureType.TABLE
        elif len(text) > 5000 and paragraph_count > 10:
            analysis["structure_type"] = StructureType.COMPLEX
        else:
            analysis["structure_type"] = StructureType.SIMPLE

        return analysis

    async def extract_hierarchical_structure(
        self, text: str
    ) -> List[HierarchicalStructure]:
        """
        Trích xuất cấu trúc phân cấp từ document

        Args:
            text: Nội dung document

        Returns:
            List các HierarchicalStructure
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._extract_hierarchical_structure_sync, text
        )

    def _extract_hierarchical_structure_sync(
        self, text: str
    ) -> List[HierarchicalStructure]:
        """Synchronous hierarchical structure extraction"""
        lines = text.split("\n")
        hierarchy: List[HierarchicalStructure] = []

        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue

            # Check each heading pattern
            for j, pattern in enumerate(self.ALL_HEADING_PATTERNS):
                match = re.match(pattern, line_stripped, re.IGNORECASE)
                if match:
                    hierarchy.append(
                        {
                            "line_number": i,
                            "level": (j % 4) + 1,  # Map to levels 1-4
                            "text": line_stripped,
                        }
                    )
                    break

        return hierarchy
