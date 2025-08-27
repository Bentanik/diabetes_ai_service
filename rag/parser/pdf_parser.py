import asyncio
import logging
import re
from pathlib import Path
from typing import List, Union

from pdfminer.high_level import extract_pages
from pdfminer.layout import LAParams, LTTextContainer, LTChar
import pdfplumber

from .base import BaseParser
from ..dataclasses import ParsedContent, FileType
from .cleaners import clean_text


class PDFParser(BaseParser):
    def __init__(self, **engine_kwargs):
        self.engine_kwargs = engine_kwargs
        self.logger = logging.getLogger(__name__)
        self._init_engine()

    def _init_engine(self):
        default_params = {
            'line_overlap': 0.5,
            'char_margin': 2.0,
            'line_margin': 0.5,
            'word_margin': 0.1,
            'boxes_flow': 0.5,
            'detect_vertical': True,
            'all_texts': False
        }
        params = {**default_params, **self.engine_kwargs}
        self.layout_params = LAParams(**params)

    def get_file_extensions(self) -> List[str]:
        return FileType.extensions[FileType.PDF]

    def get_file_type(self) -> str:
        return FileType.PDF

    async def parse_async(self, file_path: Union[str, Path]) -> ParsedContent:
        return await asyncio.to_thread(self._parse, file_path)

    def _is_heading(self, elem: LTTextContainer) -> bool:
        """Chỉ coi là heading nếu có font lớn hoặc in đậm"""
        try:
            chars = []
            for text_line in elem:
                if hasattr(text_line, '__iter__'):
                    for char in text_line:
                        if isinstance(char, LTChar):
                            chars.append(char)
                elif isinstance(char, LTChar):
                    chars.append(char)

            if not chars:
                return False

            sizes = [c.size for c in chars if c.size]
            fonts = [c.fontname for c in chars if c.fontname]

            if not sizes:
                return False

            avg_size = sum(sizes) / len(sizes)
            is_bold = any('Bold' in (font or '') or 'B' in (font or '') for font in fonts)

            return avg_size > 14 or is_bold
        except Exception:
            return False

    def _is_valid_heading_text(self, text: str) -> bool:
        """Kiểm tra xem text có phù hợp làm heading không"""
        if not text.strip():
            return False
        # Nếu quá dài → không phải heading
        if len(text) > 100:
            return False
        # Nếu chứa từ như "là một", "được", "năm" → có vẻ là nội dung
        noise_keywords = ['là một', 'được', 'sinh', 'mất', 'năm', 'là một', 'với', 'tại', 'khi']
        if any(kw in text.lower() for kw in noise_keywords):
            return False
        # Nếu có dấu chấm, phẩy dài → không phải heading
        if text.count(',') > 1 or text.count('.') > 0:
            return False
        return True

    def _is_noise_block(self, text: str) -> bool:
        """Kiểm tra có phải là block không mong muốn"""
        noise_patterns = [
            r'\[.*Đọc tiếp.*\]',
            r'Mới chọn:',
            r'Ấn phẩm:',
            r'Từ những bài viết mới của Wikipedia',
            r'Wikipedia là dự án bách khoa toàn thư mở'
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in noise_patterns)

    def _parse(self, file_path: Union[str, Path]) -> ParsedContent:
        file_path = self._validate_file(file_path)

        text_parts = []
        tables = []

        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_tables = page.extract_tables()
                    if page_tables:
                        for table in page_tables:
                            cleaned_table = [
                                [cell if cell is not None else "" for cell in row]
                                for row in table if row
                            ]
                            if cleaned_table:
                                tables.append(cleaned_table)
                                text_parts.append("[TABLE]")

            pages = extract_pages(str(file_path), laparams=self.layout_params)
            current_heading = None

            for page in pages:
                for elem in page:
                    if isinstance(elem, LTTextContainer):
                        text = elem.get_text().strip()
                        if not text:
                            continue

                        # Làm sạch sơ bộ để kiểm tra
                        clean_for_check = re.sub(r'\s+', ' ', text).strip()

                        # Kiểm tra có phải noise
                        if self._is_noise_block(clean_for_check):
                            continue

                        # Phát hiện heading
                        if self._is_heading(elem) and self._is_valid_heading_text(text):
                            # Nếu có heading cũ đang chờ, gán nó
                            if current_heading:
                                text_parts.append(f"[HEADING]{current_heading}[/HEADING]")
                            current_heading = text
                        else:
                            # Đây là nội dung → gán vào sau heading
                            if current_heading:
                                text_parts.append(f"[HEADING]{current_heading}[/HEADING]")
                                current_heading = None
                            text_parts.append(text)

            # Đóng heading cuối nếu còn
            if current_heading:
                text_parts.append(f"[HEADING]{current_heading}[/HEADING]")

        except Exception as e:
            self.logger.error(f"Parse failed: {e}")
            raise

        full_text = "\n".join(filter(None, text_parts))
        full_text = clean_text(full_text)  # Làm sạch cuối cùng

        metadata = {
            'file_name': file_path.name,
            'file_size': file_path.stat().st_size,
            'parser': 'pdfminer+pdfplumber'
        }

        return ParsedContent(
            content=full_text,
            metadata=metadata,
            file_type=self.get_file_type(),
            file_path=str(file_path),
            tables=tables
        )