import asyncio
from typing import List, Union, Dict, Any
from pathlib import Path
import logging
import re

from .base import BaseParser, ParsedContent
from rag.dataclasses import FileType

from pdfminer.high_level import extract_pages, extract_text
from pdfminer.layout import LAParams, LTTextContainer


class PDFParser(BaseParser):
    """Parser cho file PDF sử dụng PDFMiner"""
    
    def __init__(self, **engine_kwargs):
        """
        Khởi tạo PDF parser
        
        Args:
            **engine_kwargs: Các tham số cho PDFMiner LAParams
        """
        self.engine_kwargs = engine_kwargs
        self.logger = logging.getLogger(__name__)
        self._init_engine()
    
    def _init_engine(self):
        """Khởi tạo PDFMiner engine"""
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
    
    def get_file_type(self) -> FileType:
        return FileType.PDF
    
    def get_file_extensions(self) -> List[str]:
        return FileType.PDF.extensions

    async def parse_async(self, file_path: Union[str, Path]) -> ParsedContent:
        return await asyncio.to_thread(self._parse, file_path)
    
    def _parse(self, file_path: Union[str, Path]) -> ParsedContent:
        """
        Parse một file PDF từ đường dẫn
        
        Args:
            file_path: Đường dẫn tới file PDF
            
        Returns:
            ParsedContent: Object chứa content và metadata
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File PDF không tồn tại: {file_path}")
            
            # Validate PDF file
            if not self._is_valid_pdf(file_path):
                raise ValueError(f"File không phải là PDF hợp lệ: {file_path}")
            
            # Parse content
            content = self._parse_with_pdfminer(file_path)
            
            # Extract metadata
            metadata = self._extract_metadata(file_path)
            
            file_type = self.get_file_type()
            
            return ParsedContent(
                content=content,
                metadata=metadata,
                file_type=file_type,
                file_path=str(file_path)
            )
        
        except Exception as e:
            self.logger.error(f"Lỗi khi parse file {file_path}: {str(e)}")
            raise Exception(f"Lỗi khi parse file {file_path}: {str(e)}") from e
    
    def _is_valid_pdf(self, file_path: Path) -> bool:
        """Kiểm tra file có phải PDF hợp lệ không"""
        try:
            with open(file_path, 'rb') as file:
                header = file.read(4)
                return header == b'%PDF'
        except Exception:
            return False
    
    def _parse_with_pdfminer(self, file_path: Path) -> str:
        """Parse PDF với PDFMiner engine"""
        try:
            pages = extract_pages(str(file_path), laparams=self.layout_params)
            
            page_texts = []
            for page_num, page in enumerate(pages, 1):
                text = ""
                for container in page:
                    if isinstance(container, LTTextContainer):
                        container_text = container.get_text().strip()
                        if container_text and len(container_text) > 1:  # Filter out single chars
                            text += container_text + "\n"
                
                # Clean up text
                text = self._clean_text(text)
                if text.strip():  # Only add non-empty pages
                    page_texts.append(text.strip())
            
            return "\f".join(page_texts)
        
        except Exception as e:
            self.logger.warning(f"PDFMiner detailed parsing failed, fallback to simple extraction: {e}")
            # Fallback to simple text extraction
            try:
                content = extract_text(str(file_path))
                return self._clean_text(content)
            except Exception as fallback_error:
                raise Exception(f"Both detailed and simple PDF parsing failed: {fallback_error}") from e
    
    def _clean_text(self, text: str) -> str:
        """Clean và chuẩn hóa text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        lines = []
        for line in text.split('\n'):
            line = line.strip()
            if line:
                lines.append(line)
        
        # Join lines with single newline
        cleaned = '\n'.join(lines)
        
        # Remove multiple consecutive newlines
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        
        return cleaned
    
    def _extract_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata từ PDF"""
        metadata = {
            'file_size': file_path.stat().st_size,
            'file_name': file_path.name,
            'parser_engine': 'pdfminer'
        }
        
        try:
            # Count pages by extracting all pages
            pages = list(extract_pages(str(file_path), maxpages=None))
            metadata['num_pages'] = len(pages)
            
        except Exception as e:
            self.logger.warning(f"Không thể extract metadata: {e}")
            metadata['num_pages'] = 'unknown'
        
        return metadata