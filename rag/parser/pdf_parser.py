import asyncio
import logging
import re
import unicodedata
from pathlib import Path
from typing import List, Union, Dict, Any

from pdfminer.high_level import extract_pages
from pdfminer.layout import LAParams, LTTextContainer, LTChar
import pdfplumber

from .base import BaseParser
from ..dataclasses import ParsedContent, FileType


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

    def _parse(self, file_path: Union[str, Path]) -> ParsedContent:
        file_path = self._validate_file(file_path)

        try:
            content, tables = self._extract_text_and_tables(file_path)
            content = self._clean_text(content)
            metadata = self._build_metadata(file_path, tables)

            assert isinstance(metadata, dict), f"metadata must be dict, got {type(metadata)}"

            return ParsedContent(
                content=content,
                metadata=metadata,
                file_type=self.get_file_type(),
                file_path=str(file_path),
                tables=tables
            )
        except Exception as e:
            self.logger.error(f"Failed to parse PDF {file_path}: {str(e)}")
            raise

    def _extract_text_and_tables(self, file_path: Path) -> tuple[str, List[List[List[str]]]]:
        text_parts = []
        tables = []

        # Extract tables
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_tables = page.extract_tables()
                    if page_tables:
                        tables.extend(page_tables)
        except Exception as e:
            self.logger.warning(f"Table extraction failed: {e}")

        # Extract text
        try:
            pages = extract_pages(str(file_path), laparams=self.layout_params)
            for page in pages:
                page_lines = []
                for elem in page:
                    if isinstance(elem, LTTextContainer):
                        text = elem.get_text().strip()
                        if not text:
                            continue

                        # Detect heading by font size
                        try:
                            # Lấy tất cả ký tự trong container
                            chars = []
                            for text_line in elem:
                                if hasattr(text_line, '__iter__'):
                                    for char in text_line:
                                        if isinstance(char, LTChar):
                                            chars.append(char)
                                elif isinstance(text_line, LTChar):
                                    chars.append(text_line)

                            if chars:
                                max_size = max(char.size for char in chars)
                                if max_size > 14:
                                    text = f"\n[HEADING]{text}[/HEADING]\n"
                        except Exception as e:
                            self.logger.debug(f"Font size detection failed for: {text[:30]}... Error: {e}")

                        page_lines.append(text)
                if page_lines:
                    text_parts.append("\n".join(page_lines))
        except Exception as e:
            self.logger.error(f"Text extraction failed: {e}")
            raise ValueError(f"Cannot extract text from PDF: {e}")

        full_text = "\f[PAGE_BREAK]\f".join(text_parts)
        return full_text, tables

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'\r\n|\r', '\n', text)
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        text = re.sub(r'([a-z,;:])\n([a-z])', r'\1 \2', text)
        lines = [line.strip() for line in text.split('\n')]
        cleaned = '\n'.join(line if line else "" for line in lines)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = unicodedata.normalize('NFC', cleaned)
        return cleaned.strip()

    def _build_metadata(self, file_path: Path, tables: List) -> Dict[str, Any]:
        try:
            with pdfplumber.open(file_path) as pdf:
                num_pages = len(pdf.pages)
        except Exception as e:
            self.logger.warning(f"Cannot get num_pages: {e}")
            num_pages = 'unknown'

        return {
            'file_name': file_path.name,
            'file_size': file_path.stat().st_size,
            'num_pages': num_pages,
            'num_tables': len(tables),
            'parser': 'pdfminer+pdfplumber'
        }