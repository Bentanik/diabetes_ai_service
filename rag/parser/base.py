from abc import ABC, abstractmethod
from typing import List, Union
from pathlib import Path

from rag.dataclasses import ParsedContent, FileType


class BaseParser(ABC):
    """
    Abstract base class cho parser
    
    Định nghĩa interface chung mà tất cả parser phải implement:
    - Xác định loại file được hỗ trợ
    - Parse file thành ParsedContent
    - Validate file trước khi parse
    """
    
    @abstractmethod
    def get_file_extensions(self) -> List[str]:
        """
        Trả về danh sách file extensions được hỗ trợ
        
        Returns:
            List[str]: Danh sách extension (ví dụ: ['.pdf', '.txt'])
        """
        pass
    
    @abstractmethod
    def get_file_type(self) -> FileType:
        """
        Trả về tên loại file mà parser này xử lý
        
        Returns:
            str: Tên loại file (ví dụ: "PDF", "DOCX", "TXT")
        """
        pass
    
    @abstractmethod
    async def parse_async(self, file_path: Union[str, Path]) -> ParsedContent:
        """
        Parse file và trả về nội dung đã được xử lý
        
        Args:
            file_path: Đường dẫn đến file cần parse
            
        Returns:
            ParsedContent: Object chứa nội dung text và metadata
            
        Raises:
            FileNotFoundError: Khi file không tồn tại
            ValueError: Khi file không được hỗ trợ hoặc không parse được
        """
        pass
    
    def can_parse(self, file_path: Union[str, Path]) -> bool:
        """
        Kiểm tra parser này có thể xử lý file hay không
        
        Args:
            file_path: Đường dẫn đến file
            
        Returns:
            bool: True nếu có thể parse, False nếu không
        """
        file_path = Path(file_path)
        return file_path.suffix.lower() in self.get_file_extensions()
    
    def _validate_file(self, file_path: Union[str, Path]) -> Path:
        """
        Validate file tồn tại và có thể được parser
        
        Args:
            file_path: Đường dẫn đến file
            
        Returns:
            Path: Path object đã được validate
            
        Raises:
            FileNotFoundError: Khi file không tồn tại
            ValueError: Khi file extension không được hỗ trợ
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not self.can_parse(file_path):
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        return file_path
