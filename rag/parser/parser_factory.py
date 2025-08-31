from pathlib import Path
from typing import List, Union, Type

from .base import BaseParser
from .pdf_parser import PDFParser
from .txt_parser import TXTParser
from .docx_parser import DocxParser
from ..dataclasses import FileType


class ParserFactory:
    """
    Factory tạo parser phù hợp dựa trên định dạng file hoặc nội dung.
    """

    _parsers: dict = {}

    @classmethod
    def _initialize_parsers(cls):
        """Khởi tạo parsers nếu chưa có"""
        if not cls._parsers:
            cls._parsers = {
                FileType.PDF: PDFParser,
                FileType.TXT: TXTParser,
                FileType.DOCX: DocxParser,
            }

    @classmethod
    def get_parser(cls, source: Union[str, Path]) -> BaseParser:
        cls._initialize_parsers()

        # Xử lý URL, query params, fragment
        path_str = str(source).split("?")[0].split("#")[0]
        path = Path(path_str)

        # Nếu là đường dẫn hợp lệ và có suffix
        if path.suffix:
            suffix = path.suffix.lower()
            for file_type, extensions in FileType.extensions.items():
                if suffix in extensions:
                    parser_class = cls._parsers.get(file_type)
                    if parser_class:
                        return parser_class()

        # Kiểm tra nếu file tồn tại nhưng không có suffix
        if path.exists() and not path.suffix:
            raise ValueError(f"Không thể xác định loại file: không có phần mở rộng ({path})")

        if not path.exists():
            raise ValueError(f"File không tồn tại hoặc không thể truy cập: {path}")

        raise ValueError(f"Không hỗ trợ định dạng file: {path.suffix} (từ {source})")

    @classmethod
    def register_parser(cls, file_type: str, parser_class: Type[BaseParser]):
        """Đăng ký parser mới cho một loại file."""
        cls._initialize_parsers()
        cls._parsers[file_type] = parser_class

    @classmethod
    def supported_extensions(cls) -> List[str]:
        """Trả về danh sách các phần mở rộng được hỗ trợ (ví dụ: ['.pdf', '.txt', '.docx'])"""
        cls._initialize_parsers()
        extensions = []
        for file_type, parser_class in cls._parsers.items():
            if file_type in FileType.extensions:
                extensions.extend(FileType.extensions[file_type])
            else:
                # Hỗ trợ custom type nếu không nằm trong FileType
                ext = f".{file_type.value}" if hasattr(file_type, 'value') else f".{file_type}"
                extensions.append(ext)
        return sorted(list(set(extensions)))