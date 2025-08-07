from dataclasses import dataclass
from .pdf.text_block import BlockMetadata

@dataclass
class PDFChunkMetadata:
    block_metadata: BlockMetadata

@dataclass
class Chunk:
    text: str
    metadata: PDFChunkMetadata
