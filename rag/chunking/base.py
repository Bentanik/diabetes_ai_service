from abc import ABC, abstractmethod
from typing import List
from rag.dataclasses import ParsedContent, DocumentChunk

class BaseChunker(ABC):
    """
    Abstract base class cho chunker.
    
    Định nghĩa interface chung mà tất cả parser phải implement:
    - Chunk text thành danh sách DocumentChunk
    """

    @abstractmethod
    async def chunk_async(self, parsed_content: ParsedContent) -> List[DocumentChunk]:
        """
        Abstract method: Chunk text thành danh sách DocumentChunk.
        
        Args:
            text (str): Văn bản đầu vào cần chunk.
            source_file (str): Đường dẫn hoặc tên file nguồn để lưu metadata.
            
        Returns:
            List[DocumentChunk]: Danh sách các DocumentChunk.
        """
        pass
