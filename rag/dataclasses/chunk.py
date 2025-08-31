from dataclasses import dataclass

@dataclass
class ChunkMetadata:
    chunk_index: int
    word_count: int
    section: str
    chunking_strategy: str


@dataclass
class DocumentChunk:
    content: str
    metadata: ChunkMetadata