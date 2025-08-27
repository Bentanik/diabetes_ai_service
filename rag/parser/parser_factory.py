from pathlib import Path
from typing import List, Union, Type

from .pdf_parser import PDFParser
from .base import BaseParser
from ..dataclasses import FileType


class ParserFactory:
    """
    Factory tạo parser phù hợp dựa trên định dạng file hoặc nội dung.
    """

    _parsers: dict = {
        FileType.PDF: PDFParser,
    }

    @classmethod
    def get_parser(cls, source: Union[str, Path]) -> BaseParser:
        path = Path(str(source).split("?")[0].split("#")[0])

        suffix = path.suffix.lower()
        if suffix in [".pdf"]:
            return PDFParser()

        if path.exists():
            if path.suffix == "":
                raise ValueError("Không thể xác định loại file: không có phần mở rộng")
            return PDFParser()

        raise ValueError(f"Không thể xác định parser cho nguồn: {source}")

    @classmethod
    def register_parser(cls, file_type: str, parser_class: Type[BaseParser]):
        cls._parsers[file_type] = parser_class

    @classmethod
    def supported_extensions(cls) -> List[str]:
        extensions = []
        for key, parser in cls._parsers.items():
            if key in FileType.extensions.values():
                extensions.extend(FileType.extensions[key])
            else:
                extensions.append(f".{key}")
        return list(set(extensions))