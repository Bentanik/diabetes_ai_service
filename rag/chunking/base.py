from abc import ABC, abstractmethod
from typing import List
from ..dataclasses import DocumentChunk, ParsedContent

class BaseChunker(ABC):
    @abstractmethod
    async def chunk_async(self, parsed_content: ParsedContent) -> List[DocumentChunk]:
        pass