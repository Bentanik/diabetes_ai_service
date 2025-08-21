from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ParsedContent:
    """a    
    Attributes:
        content: Nội dung text đã extract từ file
        metadata: Metadata chứa thông tin bổ sung (page_count, author, created_date, file_size, tags, is_encrypted, etc.)
        file_type: Loại file (pdf, docx, txt, etc.)
        file_path: Đường dẫn gốc của file
    """
    content: str                    
    metadata: Optional[Dict[str, Any]]   
    file_type: str                  
    file_path: str                  
    
    def is_empty(self) -> bool:
        """
        Kiểm tra content có rỗng không
        
        Returns:
            bool: True nếu content rỗng hoặc chỉ chứa whitespace
        """
        return not self.content.strip()
