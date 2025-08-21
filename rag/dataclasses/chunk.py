from dataclasses import dataclass

@dataclass
class ChunkMetadata:
    chunk_index: int
    word_count: int
    chunking_strategy: str


@dataclass
class DocumentChunk:
    content: str
    metadata: ChunkMetadata