import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import PreTrainedTokenizerBase

from rag.types.types import Chunk, ChunkMetadata, StructureType
from rag.types.pdf_types import TextBlock
from rag.types import LanguageInfo
from rag.types.chunking_config import ChunkingConfig
from rag.common.structure_analyzer import DocumentStructureAnalyzer
from rag.common.language_detector import LanguageDetector


class Chunking:
    def __init__(
        self,
        config: Optional["ChunkingConfig"] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ):
        self.config = config if config is not None else ChunkingConfig()
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.structure_analyzer = DocumentStructureAnalyzer()
        self.lang_detector = LanguageDetector()
        self._init_splitters()

        if self.tokenizer is None:
            raise ValueError("Phải truyền tokenizer vào cho Chunking!")

    def _init_splitters(self) -> None:
        separators = [
            "\n\n\n",
            "\n\n",
            "\n",
            "。",
            "．",
            "！",
            "？",
            "；",
            ".",
            "!",
            "?",
            ";",
            ":",
            " ",
            "",
        ]
        self.splitter = RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=self.config.max_chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=self._count_tokens,
            is_separator_regex=False,
        )

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    async def _chunk_semantic(self, block: TextBlock) -> List[Chunk]:
        import re

        text = block.context.strip()
        paragraphs = re.split(r"\n\s*\n", text)
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_tokens = 0

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            para_tokens = self._count_tokens(paragraph)
            # Nếu paragraph quá dài, cần cắt nhỏ thêm
            if para_tokens > self.config.max_chunk_size:
                # Nếu đang có chunk, đóng lại trước khi thêm các chunk nhỏ
                if current_chunk:
                    chunk_content = "\n\n".join(current_chunk)
                    chunks.append(chunk_content)
                    current_chunk = []
                    current_tokens = 0
                # Cắt paragraph thành các đoạn nhỏ hơn max_chunk_size
                tokens = self.tokenizer.encode(paragraph)
                for i in range(0, len(tokens), self.config.max_chunk_size):
                    chunk_tokens = tokens[i : i + self.config.max_chunk_size]
                    chunk_text = self.tokenizer.decode(chunk_tokens)
                    chunks.append(chunk_text)
                continue

            # Nếu cộng paragraph vào chunk hiện tại mà vượt max_chunk_size, đóng chunk hiện tại
            if (
                current_tokens + para_tokens > self.config.max_chunk_size
                and current_chunk
            ):
                chunk_content = "\n\n".join(current_chunk)
                chunks.append(chunk_content)
                current_chunk = [paragraph]
                current_tokens = para_tokens
            else:
                current_chunk.append(paragraph)
                current_tokens += para_tokens

        # Đóng chunk cuối cùng nếu còn
        if current_chunk:
            chunk_content = "\n\n".join(current_chunk)
            chunks.append(chunk_content)

        results: List[Chunk] = []
        for i, chunk in enumerate(chunks):
            lang_info: LanguageInfo = self.lang_detector.detect_language(chunk)
            tokens = self._count_tokens(chunk)
            chunk_metadata: ChunkMetadata = {
                "strategy": "semantic",
                "token_count": tokens,
                "language_info": lang_info,
                "source": block.block_id,
                "block_metadata": block.metadata,
            }
            results.append(
                {
                    "text": chunk,
                    "metadata": chunk_metadata,
                }
            )
        return results

    async def chunk_text(self, blocks: List[TextBlock]) -> List[Chunk]:
        """Chunk list các TextBlock, trả về list Chunk chuẩn type"""
        all_chunks: List[Chunk] = []
        for idx, block in enumerate(blocks):
            text = block.context.strip()
            if not text:
                continue
            # Phân tích cấu trúc cho block nếu chưa có
            if not block.metadata.structure_analysis:
                block.metadata.structure_analysis = (
                    await self.structure_analyzer.analyze_structure(text)
                )
            structure_type = block.metadata.structure_analysis.get(
                "structure_type", StructureType.SIMPLE
            )
            token_count = self._count_tokens(text)
            if token_count < self.config.min_chunk_size:
                lang_info: LanguageInfo = self.lang_detector.detect_language(text)
                chunk_metadata: ChunkMetadata = {
                    "strategy": "short",
                    "token_count": token_count,
                    "language_info": lang_info,
                    "source": block.block_id,
                    "structure_analysis": block.metadata.structure_analysis,
                }
                all_chunks.append(
                    {
                        "text": text,
                        "metadata": chunk_metadata,
                    }
                )
                continue
            # Chọn strategy
            # if structure_type == StructureType.HIERARCHICAL:
            #     chunks = await self._chunk_hierarchical(block, idx)
            # elif structure_type in [StructureType.COMPLEX, StructureType.TABLE]:
            #     chunks = await self._chunk_semantic(block, idx)
            # else:
            #     chunks = await self._chunk_simple(block, idx)
            chunks = await self._chunk_semantic(block)
            all_chunks.extend(chunks)
        return all_chunks


import asyncio
from transformers import AutoTokenizer

# Giả sử các class đã import từ file của bạn:
# - Chunking, ChunkingConfig, TextBlock, BlockMetadata, PageSize, LanguageInfo, StructureAnalysis


def build_textblock_from_text(text: str, block_id: str = "block_1") -> TextBlock:
    from rag.types.pdf_types.text_block import BlockMetadata

    # Tạo metadata tối thiểu
    bbox = (0, 0, 100, 100)  # Dummy bbox
    metadata = BlockMetadata(
        bbox=bbox,
        block_type="paragraph",
        num_lines=text.count("\n") + 1,
        num_spans=1,
        is_cleaned=True,
        page_index=0,
        language_info=None,
        structure_analysis=None,
    )
    return TextBlock(
        block_id=block_id,
        context=text,
        metadata=metadata,
    )


async def main():
    # 1. Tạo tokenizer (thay bằng model bạn dùng, ví dụ "vinai/phobert-base" hoặc "bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # 2. Đoạn text mẫu
    text = """Text Block Models - Data structures cho text blocks từ PDF

Module chứa các dataclass và type definitions cho việc trích xuất text từ PDF:
- BlockMetadata: Metadata của một text block
- TextBlock: Một text block hoàn chỉnh với content và metadata  
- PageSize: Kích thước trang PDF
- PageData: Data của một trang PDF hoàn chỉnh
"""

    # 3. Tạo TextBlock
    block = build_textblock_from_text(text)

    # 4. Tạo config và Chunking
    config = ChunkingConfig(max_chunk_size=64, chunk_overlap=16, min_chunk_size=10)
    chunker = Chunking(tokenizer=tokenizer, config=config)

    # 5. Chunk text
    chunks = await chunker.chunk_text([block])

    # 6. In kết quả
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}:")
        print("Text:", chunk["text"])
        print("Metadata:", chunk["metadata"])
        print("-" * 40)


if __name__ == "__main__":
    asyncio.run(main())
