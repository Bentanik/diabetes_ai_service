# rag/chunkers/semantic_chunker.py
import re
import numpy as np
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from transformers import PreTrainedTokenizerFast

from rag.dataclasses import ParsedContent

from .base import BaseChunker
from ..dataclasses import DocumentChunk, ChunkMetadata

class Chunker(BaseChunker):
    """
    Hỗ trợ:
    - Tiếng Việt và tiếng Anh
    - Xử lý bảng (được chèn dưới dạng [TABLE])
    - Tách chunk tại điểm thay đổi chủ đề
    - Giữ nguyên câu và cấu trúc (heading, page break)
    - Dùng token count chính xác
    """

    def __init__(
        self,
        embedding_model,
        tokenizer: PreTrainedTokenizerFast = None,
        max_tokens: int = 512,
        min_tokens: int = 100,
        overlap_tokens: int = 64,
        similarity_threshold: float = 0.6,
    ):
        """
        Args:
            embedding_model: Instance của EmbeddingModel (singleton)
            tokenizer: Tokenizer tương ứng (nếu không truyền, tự load từ model)
            max_tokens: Số token tối đa mỗi chunk
            min_tokens: Số token tối thiểu để tạo chunk
            overlap_tokens: Số token overlap giữa các chunk
            similarity_threshold: Ngưỡng similarity để coi là "thay đổi chủ đề"
        """
        self.embedding_model = embedding_model
        self.tokenizer = tokenizer or embedding_model.model.tokenizer
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.overlap_tokens = overlap_tokens
        self.similarity_threshold = similarity_threshold

        # Setup sentence tokenizer cho tiếng Việt
        try:
            from underthesea import sent_tokenize as viet_sent_tokenize
            self.viet_sent_tokenize = viet_sent_tokenize
        except ImportError:
            self.viet_sent_tokenize = None

    def _count_tokens(self, text: str) -> int:
        """Đếm số subword token"""
        return len(self.tokenizer.encode(text))

    def _split_sentences(self, text: str) -> List[str]:
        """Tách câu, tự động detect ngôn ngữ"""
        if not text or not text.strip():
            return []
        text = re.sub(r'\s+', ' ', text).strip()

        # Detect language
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
        """Chuyển bảng thành markdown table"""
        if not table or not table[0]:
            return ""
        try:
            header = "| " + " | ".join(table[0]) + " |"
            separator = "| " + " | ".join(["---"] * len(table[0])) + " |"
            rows = [
                "| " + " | ".join(row) + " |"
                for row in table[1:] if row
            ]
            return "\n".join([header, separator] + rows)
        except Exception:
            return "\n".join(["\t".join(row) for row in table if row])

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

    def _split_by_structure(self, text: str) -> List[str]:
        """
        Tách text theo cấu trúc: heading, page break
        Trả về danh sách các block
        """
        # Tách theo heading và page break
        parts = re.split(r'(\[PAGE_BREAK\]|\[HEADING\][^\[]*\[/HEADING\])', text)
        blocks = []
        current_block = ""

        for part in parts:
            part = part.strip()
            if not part:
                continue

            if re.match(r'\[PAGE_BREAK\]|\[HEADING\]', part):
                # Đóng block hiện tại
                if current_block:
                    blocks.append(current_block.strip())
                    current_block = ""
                # Thêm block cấu trúc
                blocks.append(part)
            else:
                current_block += " " + part

        if current_block.strip():
            blocks.append(current_block.strip())

        return blocks

    def _get_overlap_text(self, text: str, max_tokens: int) -> str:
        """Lấy overlap từ cuối text"""
        sentences = self._split_sentences(text)
        overlap = ""
        for sent in reversed(sentences):
            candidate = sent + " " + overlap
            if self._count_tokens(candidate) > max_tokens:
                break
            overlap = candidate
        return overlap.strip()

    def _make_chunk(self, content: str, chunk_index: int) -> DocumentChunk:
        """Tạo DocumentChunk"""
        return DocumentChunk(
            content=content.strip(),
            metadata=ChunkMetadata(
                chunk_index=chunk_index,
                word_count=len(content.split()),
                chunking_strategy="semantic_chunk"
            )
        )

    async def _embed_sentences(self, sentences: List[str]) -> np.ndarray:
        """Embed danh sách câu bằng EmbeddingModel"""
        if not sentences:
            return np.array([])
        try:
            embeddings = await self.embedding_model.embed_batch(sentences)
            return np.array(embeddings)
        except Exception as e:
            raise RuntimeError(f"Embedding failed: {e}")

    async def _find_semantic_breaks(self, sentences: List[str]) -> List[int]:
        """Tìm điểm ngắt ngữ nghĩa dựa trên similarity"""
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
        Chia parsed_content thành các chunk ngữ nghĩa.

        Args:
            parsed_content: Output từ parser (có .content, .tables)

        Returns:
            List[DocumentChunk]
        """
        text = parsed_content.content
        tables = parsed_content.tables or []

        # === BƯỚC 1: Inject bảng vào text ===
        if tables:
            # Đảm bảo có placeholder
            text = re.sub(r'\[TABLE_PLACEHOLDER\]', '[TABLE]', text)
            text = self._inject_tables(text, tables)

        # === BƯỚC 2: Tách theo cấu trúc (heading, page break) ===
        blocks = self._split_by_structure(text)

        # === BƯỚC 3: Chunk từng block ===
        chunks = []
        current_chunk = ""
        current_token_count = 0
        chunk_index = 0

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            if block == "[PAGE_BREAK]":
                if current_chunk and current_token_count >= self.min_tokens:
                    chunks.append(self._make_chunk(current_chunk, chunk_index))
                    chunk_index += 1
                    current_chunk = ""
                    current_token_count = 0
                continue

            if block.startswith("[HEADING]") and block.endswith("[/HEADING]"):
                # Tạo chunk riêng cho heading
                if current_chunk:
                    chunks.append(self._make_chunk(current_chunk, chunk_index))
                    chunk_index += 1
                    current_chunk = ""
                    current_token_count = 0
                # Heading thành 1 chunk riêng
                chunks.append(self._make_chunk(block, chunk_index))
                chunk_index += 1
                continue

            # === Xử lý nội dung thường ===
            sentences = self._split_sentences(block)
            if not sentences:
                continue

            # Tìm điểm ngắt ngữ nghĩa
            semantic_breaks = await self._find_semantic_breaks(sentences)
            split_points = sorted(set(semantic_breaks + [len(sentences)]))

            start = 0
            for end in split_points:
                segment = " ".join(sentences[start:end])
                segment_token_count = self._count_tokens(segment)

                # Kiểm tra nếu thêm vào sẽ vượt quá giới hạn
                if current_chunk and current_token_count + segment_token_count > self.max_tokens:
                    # Đóng chunk hiện tại
                    if current_token_count >= self.min_tokens:
                        chunks.append(self._make_chunk(current_chunk, chunk_index))
                        chunk_index += 1

                        # Overlap: lấy phần cuối của chunk trước
                        overlap_text = self._get_overlap_text(current_chunk, self.overlap_tokens)
                        current_chunk = overlap_text + " " + segment
                        current_token_count = self._count_tokens(current_chunk)
                    else:
                        # Nếu chunk hiện tại quá nhỏ, thay bằng segment mới
                        current_chunk = segment
                        current_token_count = segment_token_count
                else:
                    # Thêm vào chunk hiện tại
                    if current_chunk:
                        current_chunk += " " + segment
                    else:
                        current_chunk = segment
                    current_token_count += segment_token_count

                start = end

        # Đóng chunk cuối cùng
        if current_chunk and self._count_tokens(current_chunk) >= self.min_tokens:
            chunks.append(self._make_chunk(current_chunk, chunk_index))

        return chunks