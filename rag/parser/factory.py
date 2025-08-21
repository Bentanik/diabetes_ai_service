from typing import Union
from pathlib import Path

from .base import BaseParser, ParsedContent
from .pdf_parser import PDFParser


class ParserFactory:
    """
    Factory class để tạo parser cho các loại file khác nhau
    """

    _parsers = {
        'pdf': PDFParser,
    }
    
    @classmethod
    async def parse_file_async(cls, file_path: Union[str, Path]) -> ParsedContent:
        """
        Args:
            file_path: Đường dẫn đến file cần parse
        Returns:
            ParsedContent: Object chứa nội dung và metadata đã parse
        Raises:
            ValueError: Nếu file extension không được support
        """
        parser = cls._get_parser(file_path)
        return await parser.parse_async(file_path)
    
    @classmethod
    def _get_parser(cls, file_path: Union[str, Path]) -> BaseParser:
        """
        Args:
            file_path: Đường dẫn file để xác định loại parser
        Returns:
            BaseParser: Parser instance phù hợp với file type
        """
        extension = Path(file_path).suffix.lower()
        
        extension_map = {
            '.pdf': 'pdf',
        }
        
        parser_type = extension_map.get(extension)
        if not parser_type:
            raise ValueError(f"Unsupported file extension: {extension}")
        
        parser_class = cls._parsers.get(parser_type)
        if not parser_class:
            raise ValueError(f"Parser not available for type: {parser_type}")
        
        return parser_class()


async def get_parser(file_path: Union[str, Path]) -> BaseParser:
    """
    Async-safe: chỉ lấy instance của Parser
    """
    return ParserFactory._get_parser(file_path)


async def parse_file(file_path: Union[str, Path]) -> ParsedContent:
    """
    Async-safe: parse file và trả về ParsedContent
    """
    return await ParserFactory.parse_file_async(file_path)