from dataclasses import dataclass


@dataclass
class ChunkingConfig:
    """
    Cấu hình cho quá trình chunking văn bản.
    """

    max_chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_size: int = 50
    keep_small_chunks: bool = True
