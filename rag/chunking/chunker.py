import re
import numpy as np
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity

from rag.dataclasses import ParsedContent, DocumentChunk, ChunkMetadata
from rag.parser.cleaners import is_noisy_chunk
from .base import BaseChunker


class Chunker(BaseChunker):
    def __init__(
        self,
        embedding_model,
        max_tokens: int = 512,
        min_tokens: int = 100,
        overlap_tokens: int = 64,
        similarity_threshold: float = 0.6,
    ):
        """
        Chunker chia nhỏ văn bản theo ngữ nghĩa, hỗ trợ tiếng Việt.
        """
        self.embedding_model = embedding_model
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.overlap_tokens = overlap_tokens
        self.similarity_threshold = similarity_threshold

        # Hỗ trợ tách câu tiếng Việt
        try:
            from underthesea import sent_tokenize as viet_sent_tokenize
            self.viet_sent_tokenize = viet_sent_tokenize
        except ImportError:
            self.viet_sent_tokenize = None

    def _count_tokens(self, text: str) -> int:
        """Đếm số token bằng tokenizer của embedding model"""
        return len(self.embedding_model.model.tokenizer.encode(text))

    def _split_sentences(self, text: str) -> List[str]:
        """Tách câu, tự động nhận diện tiếng Việt"""
        if not text.strip():
            return []
        text = re.sub(r'\s+', ' ', text).strip()

        # Detect tiếng Việt
        vi_chars = set('àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ')
        sample = ''.join(filter(str.isalpha, text.lower()[:100]))
        is_vietnamese = any(c in vi_chars for c in sample)

        if is_vietnamese and self.viet_sent_tokenize:
            sentences = self.viet_sent_tokenize(text)
        else:
            import nltk
            try:
                sentences = nltk.sent_tokenize(text)
            except LookupError:
                nltk.download('punkt')
                sentences = nltk.sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]

    def _table_to_markdown(self, table: List[List[str]]) -> str:
        """Chuyển bảng thành markdown"""
        if not table or not table[0]:
            return ""
        try:
            cleaned_table = []
            for row in table:
                if not row:
                    continue
                cleaned_row = [cell if cell is not None else "" for cell in row]
                cleaned_table.append(cleaned_row)

            if not cleaned_table or not cleaned_table[0]:
                return ""

            header = "| " + " | ".join(cleaned_table[0]) + " |"
            separator = "| " + " | ".join(["---"] * len(cleaned_table[0])) + " |"
            rows = [
                "| " + " | ".join(row) + " |"
                for row in cleaned_table[1:] if row
            ]
            return "\n".join([header, separator] + rows)
        except Exception:
            # Fallback
            try:
                lines = []
                for row in table:
                    if row:
                        cleaned_row = [cell if cell is not None else "" for cell in row]
                        lines.append("\t".join(cleaned_row))
                return "\n".join(lines)
            except:
                return "[Bảng không hợp lệ]"

    def _inject_tables(self, text: str, tables: List[List[List[str]]]) -> str:
        """Chèn bảng vào đúng vị trí [TABLE]"""
        if not tables:
            return text

        table_iter = iter(tables)
        while '[TABLE]' in text:
            try:
                table = next(table_iter)
                md_table = self._table_to_markdown(table)
                text = text.replace('[TABLE]', md_table, 1)
            except StopIteration:
                break
        return text

    def _split_by_structure(self, text: str) -> List[Tuple[str, str]]:
        """
        Tách theo cấu trúc: heading, subheading, page break
        Trả về list (block_type, content)
        """
        parts = re.split(r'(\[PAGE_BREAK\]|\[HEADING\][^\[]*\[/HEADING\]|\[SUBHEADING\][^\[]*\[/SUBHEADING\])', text)
        blocks = []
        current_block = ""

        for part in parts:
            part = part.strip()
            if not part:
                continue

            if part == "[PAGE_BREAK]":
                if current_block:
                    blocks.append(("paragraph", current_block))
                    current_block = ""
                blocks.append(("page_break", ""))
            elif part.startswith("[HEADING]"):
                if current_block:
                    blocks.append(("paragraph", current_block))
                    current_block = ""
                blocks.append(("heading", part))
            elif part.startswith("[SUBHEADING]"):
                if current_block:
                    blocks.append(("paragraph", current_block))
                    current_block = ""
                blocks.append(("subheading", part))
            else:
                current_block += " " + part

        if current_block:
            blocks.append(("paragraph", current_block))

        return blocks

    def _get_overlap_text(self, text: str, max_tokens: int) -> str:
        """Lấy phần overlap từ cuối"""
        sentences = self._split_sentences(text)
        overlap = ""
        for sent in reversed(sentences):
            candidate = sent + " " + overlap
            if self._count_tokens(candidate) > max_tokens:
                break
            overlap = candidate
        return overlap.strip()

    def _make_chunk(self, content: str, chunk_index: int, chunk_type: str = "content") -> DocumentChunk:
        """Tạo DocumentChunk"""
        if not content or is_noisy_chunk(content):
            return None
        return DocumentChunk(
            content=content.strip(),
            metadata=ChunkMetadata(
                chunk_index=chunk_index,
                word_count=len(content.split()),
                chunking_strategy="semantic_chunk",
            )
        )

    async def _embed_sentences(self, sentences: List[str]) -> np.ndarray:
        """Embed danh sách câu"""
        if not sentences:
            return np.array([])
        try:
            embeddings = await self.embedding_model.embed_batch(sentences)
            return np.array(embeddings)
        except Exception as e:
            raise RuntimeError(f"Embedding failed: {e}")

    async def _find_semantic_breaks(self, sentences: List[str]) -> List[int]:
        """Tìm điểm ngắt ngữ nghĩa"""
        if len(sentences) < 2:
            return []

        embeddings = await self._embed_sentences(sentences)
        breaks = []
        for i in range(1, len(embeddings)):
            sim = cosine_similarity([embeddings[i-1]], [embeddings[i]])[0][0]
            if sim < self.similarity_threshold:
                breaks.append(i)
        return breaks

    async def chunk_async(self, parsed_content: ParsedContent) -> List[DocumentChunk]:
        """
        Chia nhỏ nội dung theo ngữ nghĩa.
        """
        text = parsed_content.content
        tables = parsed_content.tables or []

        # Bước 1: Inject bảng
        if tables:
            text = self._inject_tables(text, tables)

        # Bước 2: Tách theo cấu trúc
        blocks = self._split_by_structure(text)

        # Bước 3: Chunk từng block
        chunks = []
        current_chunk = ""
        current_token_count = 0
        chunk_index = 0

        for block_type, content in blocks:
            if block_type == "page_break":
                if current_chunk and current_token_count >= self.min_tokens:
                    chunk = self._make_chunk(current_chunk, chunk_index, "content")
                    if chunk:
                        chunks.append(chunk)
                        chunk_index += 1
                    current_chunk = ""
                    current_token_count = 0
                continue

            if block_type in ["heading", "subheading"]:
                if current_chunk:
                    chunk = self._make_chunk(current_chunk, chunk_index, "content")
                    if chunk:
                        chunks.append(chunk)
                        chunk_index += 1
                    current_chunk = ""
                    current_token_count = 0
                heading_chunk = self._make_chunk(content, chunk_index, block_type)
                if heading_chunk:
                    chunks.append(heading_chunk)
                    chunk_index += 1
                continue

            # Xử lý đoạn văn
            sentences = self._split_sentences(content)
            if not sentences:
                continue

            semantic_breaks = await self._find_semantic_breaks(sentences)
            split_points = sorted(set(semantic_breaks + [len(sentences)]))

            start = 0
            for end in split_points:
                segment = " ".join(sentences[start:end])
                seg_token_count = self._count_tokens(segment)

                if current_chunk and current_token_count + seg_token_count > self.max_tokens:
                    if current_token_count >= self.min_tokens:
                        chunk = self._make_chunk(current_chunk, chunk_index, "content")
                        if chunk:
                            chunks.append(chunk)
                            chunk_index += 1
                        overlap = self._get_overlap_text(current_chunk, self.overlap_tokens)
                        current_chunk = overlap + " " + segment
                        current_token_count = self._count_tokens(current_chunk)
                    else:
                        current_chunk = segment
                        current_token_count = seg_token_count
                else:
                    current_chunk = (current_chunk + " " + segment) if current_chunk else segment
                    current_token_count += seg_token_count
                start = end

        # Đóng chunk cuối
        if current_chunk and self._count_tokens(current_chunk) >= self.min_tokens:
            chunk = self._make_chunk(current_chunk, chunk_index, "content")
            if chunk:
                chunks.append(chunk)

        return chunks