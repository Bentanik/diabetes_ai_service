import re
import numpy as np
from typing import List, Tuple

from sklearn.metrics.pairwise import cosine_similarity
from underthesea import sent_tokenize, word_tokenize
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
        self.embedding_model = embedding_model
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.overlap_tokens = overlap_tokens
        self.similarity_threshold = similarity_threshold

        self.sent_tokenize = sent_tokenize
        self.word_tokenize = word_tokenize

        self.medical_terms = {
            "đái tháo đường", "bệnh tiểu đường", "tăng huyết áp", "kháng insulin",
            "rối loạn chuyển hóa", "hội chứng buồng trứng đa nang", "tăng triglyceride",
            "tiểu đường thai kỳ", "huyết áp tâm thu", "huyết áp tâm trương",
            "chỉ số khối cơ thể", "béo phì", "mỡ máu", "đường huyết",
            "insulin đề kháng", "tổn thương thần kinh", "bệnh võng mạc",
        }

    def _count_tokens(self, text: str) -> int:
        return len(self.embedding_model.model.tokenizer.encode(text))

    def _split_sentences(self, text: str) -> List[str]:
        """Tách câu với hỗ trợ tiếng Việt"""
        text = re.sub(r'\s+', ' ', text).strip()
        if not text:
            return []
        try:
            sentences = self.sent_tokenize(text)
            return [s.strip() for s in sentences if len(s.strip()) > 3]
        except:
            return [text]

    def _table_to_markdown(self, table: List[List[str]]) -> str:
        if not table or not table[0]:
            return ""
        try:
            header = "| " + " | ".join(table[0]) + " |"
            separator = "| " + " | ".join(["---"] * len(table[0])) + " |"
            rows = ["| " + " | ".join(row) + " |" for row in table[1:] if row]
            return "\n".join([header, separator] + rows)
        except Exception:
            return "[BẢNG]"

    def _inject_tables(self, text: str, tables: List[List[List[str]]]) -> str:
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
        """Tách theo cấu trúc: heading, subheading, page break"""
        parts = re.split(
            r'(\[PAGE_BREAK\]|\[HEADING\][^\[]*\[/HEADING\])',
            text
        )
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
                heading_text = re.sub(r'\[/?HEADING\]', '', part).strip()
                blocks.append(("heading", heading_text))
            else:
                current_block += " " + part

        if current_block:
            blocks.append(("paragraph", current_block))
        return blocks

    def _get_overlap_text(self, sentences: List[str], max_tokens: int) -> str:
        """Lấy overlap là các câu cuối, không ngắt giữa câu"""
        overlap = ""
        for sent in reversed(sentences):
            candidate = sent + " " + overlap
            if self._count_tokens(candidate) > max_tokens:
                break
            overlap = candidate
        return overlap.strip()

    def _should_prevent_break(self, prev_end: str, next_start: str) -> bool:
        """Kiểm tra xem có đang ngắt giữa cụm từ y khoa không"""
        prev_end = prev_end.lower()
        next_start = next_start.lower()
        for term in self.medical_terms:
            for i in range(3, len(term) - 2):
                if prev_end.endswith(term[:i]) and next_start.startswith(term[i:]):
                    return True
        return False

    async def _embed_sentences(self, sentences: List[str]) -> np.ndarray:
        if not sentences:
            return np.array([])
        try:
            embeddings = await self.embedding_model.embed_batch(sentences)
            return np.array(embeddings)
        except Exception as e:
            raise RuntimeError(f"Embedding failed: {e}")

    async def _find_semantic_breaks(self, sentences: List[str]) -> List[int]:
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
        text = parsed_content.content
        tables = parsed_content.tables or []

        # Bước 1: Inject bảng
        text = self._inject_tables(text, tables)

        # Bước 2: Tách theo cấu trúc
        blocks = self._split_by_structure(text)

        chunks = []
        current_chunk = ""
        current_token_count = 0
        chunk_index = 0
        current_section = "Unknown"

        for block_type, content in blocks:
            if block_type == "page_break":
                if current_chunk and current_token_count >= self.min_tokens:
                    chunk = DocumentChunk(
                        content=current_chunk.strip(),
                        metadata=ChunkMetadata(
                            chunk_index=chunk_index,
                            word_count=len(current_chunk.split()),
                            section=current_section,
                            chunking_strategy="semantic_chunk",
                        )
                    )
                    if not is_noisy_chunk(chunk.content):
                        chunks.append(chunk)
                        chunk_index += 1
                    current_chunk = ""
                    current_token_count = 0
                continue

            if block_type == "heading":
                current_section = content
                # Gắn heading vào chunk tiếp theo
                continue

            # Xử lý đoạn văn
            sentences = self._split_sentences(content)
            if not sentences:
                continue

            # Tìm điểm ngắt ngữ nghĩa
            semantic_breaks = await self._find_semantic_breaks(sentences)
            split_points = sorted(set(semantic_breaks + [len(sentences)]))

            start = 0
            for end in split_points:
                segment = " ".join(sentences[start:end]).strip()
                if not segment:
                    start = end
                    continue

                seg_token_count = self._count_tokens(segment)

                # Kiểm tra ngắt giữa cụm từ
                if current_chunk and seg_token_count < 50:
                    last_sentence = current_chunk.split(" ")[-5:]
                    if self._should_prevent_break(" ".join(last_sentence), segment):
                        current_chunk += " " + segment
                        current_token_count += seg_token_count
                        start = end
                        continue

                # Kiểm tra kích thước
                if current_token_count + seg_token_count > self.max_tokens:
                    if current_token_count >= self.min_tokens:
                        # Đóng chunk hiện tại
                        chunk = DocumentChunk(
                            content=current_chunk.strip(),
                            metadata=ChunkMetadata(
                                chunk_index=chunk_index,
                                word_count=len(current_chunk.split()),
                                section=current_section,
                                chunking_strategy="semantic_chunk",
                            )
                        )
                        if not is_noisy_chunk(chunk.content):
                            chunks.append(chunk)
                            chunk_index += 1

                        # Overlap: lấy 1–2 câu cuối
                        overlap_sentences = self._split_sentences(current_chunk)[-2:]
                        overlap = self._get_overlap_text(overlap_sentences, self.overlap_tokens)

                        current_chunk = overlap + " " + segment if overlap else segment
                        current_token_count = self._count_tokens(current_chunk)
                    else:
                        # Nếu chưa đủ min, gộp thêm
                        current_chunk = (current_chunk + " " + segment) if current_chunk else segment
                        current_token_count += seg_token_count
                else:
                    current_chunk = (current_chunk + " " + segment) if current_chunk else segment
                    current_token_count += seg_token_count

                start = end

        # Đóng chunk cuối
        if current_chunk and current_token_count >= self.min_tokens:
            chunk = DocumentChunk(
                content=current_chunk.strip(),
                metadata=ChunkMetadata(
                    chunk_index=chunk_index,
                    word_count=len(current_chunk.split()),
                    section=current_section,
                    chunking_strategy="semantic_chunk",
                )
            )
            if not is_noisy_chunk(chunk.content):
                chunks.append(chunk)

        return chunks