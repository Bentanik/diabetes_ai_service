import asyncio
import logging
from pathlib import Path
from typing import List, Union

from ..dataclasses import ParsedContent, FileType
from .base import BaseParser
from .cleaners import clean_text


class TXTParser(BaseParser):
    def __init__(self, encoding: str = 'utf-8', **kwargs):
        self.encoding = encoding
        self.logger = logging.getLogger(__name__)

    def get_file_extensions(self) -> List[str]:
        return FileType.extensions[FileType.TXT]

    def get_file_type(self) -> str:
        return FileType.TXT

    async def parse_async(self, file_path: Union[str, Path]) -> ParsedContent:
        return await asyncio.to_thread(self._parse, file_path)

    def _parse(self, file_path: Union[str, Path]) -> ParsedContent:
        file_path = self._validate_file(file_path)

        try:
            with open(file_path, 'r', encoding=self.encoding) as f:
                content = f.read()
        except UnicodeDecodeError:
            # Thử với encoding khác nếu UTF-8 thất bại
            try:
                with open(file_path, 'r', encoding='latin1') as f:
                    content = f.read()
            except Exception as e:
                self.logger.error(f"Cannot read TXT file with utf-8 or latin1: {e}")
                raise

        cleaned_content = clean_text(content)
        lines = cleaned_content.split('\n')
        text_parts = []

        current_heading = None
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Heuristic đơn giản: dòng ngắn, không dấu câu → có thể là heading
            is_heading = len(line) < 100 and line == line.upper() or line.endswith(':')

            if is_heading:
                if current_heading:
                    text_parts.append(f"[HEADING]{current_heading}[/HEADING]")
                current_heading = line
            else:
                if current_heading:
                    text_parts.append(f"[HEADING]{current_heading}[/HEADING]")
                    current_heading = None
                text_parts.append(line)

        if current_heading:
            text_parts.append(f"[HEADING]{current_heading}[/HEADING]")

        final_text = "\n".join(text_parts)

        metadata = {
            'file_name': file_path.name,
            'file_size': file_path.stat().st_size,
            'parser': 'txt_parser',
            'encoding': self.encoding
        }

        return ParsedContent(
            content=final_text,
            metadata=metadata,
            file_type=self.get_file_type(),
            file_path=str(file_path),
            tables=[]
        )