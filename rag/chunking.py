import asyncio
import logging
import os
import re
import time
from typing import List, Optional, Dict, Any, Awaitable, Callable
from dataclasses import dataclass
from functools import lru_cache
from collections import Counter
import hashlib

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import nltk
from nltk.corpus import stopwords
from tenacity import retry, stop_after_attempt, wait_exponential
from utils import get_logger

# Optional Vietnamese tokenizer
try:
    from underthesea import word_tokenize as vi_tokenize

    HAS_VIETNAMESE_TOKENIZER = True
except ImportError:
    HAS_VIETNAMESE_TOKENIZER = False
    vi_tokenize = None

logger = get_logger(__name__)


@dataclass
class ChunkingConfig:
    """Configuration for chunking."""

    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_size: int = 50
    max_chunk_size: int = 1024
    keep_small_chunks: bool = True
    enable_caching: bool = True


class LanguageDetector:
    """Detects language of the input text."""

    VIETNAMESE_STOPWORDS = {
        "và",
        "của",
        "cho",
        "được",
        "với",
        "các",
        "có",
        "trong",
        "đã",
        "những",
        "này",
        "về",
        "như",
        "là",
        "để",
        "theo",
        "tại",
        "từ",
        "không",
        "còn",
        "bị",
        "khi",
        "sẽ",
        "nhiều",
        "phải",
        "vì",
        "trên",
        "dưới",
        "nếu",
        "cần",
        "bởi",
        "lúc",
        "họ",
        "tôi",
        "anh",
        "chị",
        "nó",
        "một",
        "hai",
        "ba",
        "bốn",
        "năm",
        "sáu",
        "bảy",
        "tám",
        "chín",
        "mười",
        "thì",
        "đó",
        "này",
        "đây",
        "kia",
        "nào",
        "đâu",
        "sao",
        "thế",
        "mà",
        "nhưng",
        "hoặc",
        "hay",
        "nên",
        "nữa",
        "cũng",
        "chỉ",
        "lại",
        "rồi",
        "đang",
        "vào",
        "ra",
        "lên",
        "xuống",
        "qua",
        "về",
        "đến",
        "tới",
        "sau",
        "trước",
        "giữa",
        "trong",
        "ngoài",
        "dưới",
        "trên",
        "bên",
        "cạnh",
        "gần",
        "xa",
        "ở",
    }

    _english_stopwords = None
    _nltk_initialized = False

    @classmethod
    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def _ensure_nltk_data(cls):
        """Ensure NLTK stopwords are available."""
        if cls._nltk_initialized:
            return
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            logger.info("Downloading NLTK stopwords...")
            nltk.download("stopwords", quiet=True)
        cls._english_stopwords = set(stopwords.words("english"))
        cls._nltk_initialized = True

    @staticmethod
    @lru_cache(maxsize=1000)
    def detect_language(text: str) -> Dict[str, Any]:
        """Detect language with caching."""
        if not text:
            return {"language": "unknown", "vietnamese_ratio": 0, "confidence": 0}

        vietnamese_chars = (
            r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđĐ]"
        )
        vietnamese_count = len(re.findall(vietnamese_chars, text, re.IGNORECASE))
        total_chars = len(
            re.findall(
                r"[a-zA-ZàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđĐ]",
                text,
            )
        )

        if total_chars == 0:
            return {"language": "unknown", "vietnamese_ratio": 0, "confidence": 0}

        vietnamese_ratio = vietnamese_count / total_chars
        if vietnamese_ratio > 0.20:
            language = "vietnamese"
            confidence = min(vietnamese_ratio * 2, 1.0)
        elif vietnamese_ratio > 0.05:
            language = "mixed"
            confidence = 0.7
        else:
            language = "english"
            confidence = 1.0 - vietnamese_ratio

        return {
            "language": language,
            "vietnamese_ratio": vietnamese_ratio,
            "confidence": confidence,
        }

    @classmethod
    def tokenize_text(cls, text: str, lang_info: Dict[str, Any]) -> List[str]:
        """Tokenize text based on language."""
        cls._ensure_nltk_data()
        if lang_info["language"] == "vietnamese" and HAS_VIETNAMESE_TOKENIZER:
            try:
                tokens = vi_tokenize(text) if vi_tokenize else text.split()
                return [t for t in tokens if t.lower() not in cls.VIETNAMESE_STOPWORDS]
            except Exception as e:
                logger.warning(
                    f"Vietnamese tokenization failed: {e}, falling back to simple tokenization"
                )
        tokens = re.findall(r"\b\w+\b", text.lower())
        return [t for t in tokens if t not in cls._english_stopwords]


class StructureAnalyzer:
    """Analyzes the structure of the input text."""

    ALL_HEADING_PATTERNS = [
        r"^(CHƯƠNG|PHẦN|MỤC|BÀI|TIẾT|PHỤLỤC|PHỤ LỤC)\s+[IVXLCDM\d]+",
        r"^\d+\.\s+[A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐĐ]",
        r"^[IVXLCDM]+\.\s+[A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐĐ]",
        r"^(CHAPTER|SECTION|PART|ARTICLE|APPENDIX)\s+[IVXLCDM\d]+",
        r"^\d+\.\s+[A-Z]",
        r"^[IVXLCDM]+\.\s+[A-Z]",
        r"^#{1,6}\s+",
        r"^[A-Z][A-Z\s]{2,}$",
        r"^[A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐĐ][A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐĐ\s]{2,}$",
    ]

    LIST_PATTERNS = [
        r"^[-•\*]\s+",
        r"^\d+[\.\)]\s+",
        r"^[a-zA-Z][\.\)]\s+",
        r"^[IVXLCDM]+[\.\)]\s+",
    ]

    @staticmethod
    @lru_cache(maxsize=500)
    def analyze_structure(text: str) -> Dict[str, Any]:
        """Analyze text structure with caching."""
        if not text:
            return {
                "has_headings": False,
                "has_lists": False,
                "has_tables": False,
                "has_code": False,
                "paragraph_count": 0,
                "structure_type": "simple",
            }

        lines = text.split("\n")
        analysis = {
            "has_headings": False,
            "has_lists": False,
            "has_tables": False,
            "has_code": False,
            "paragraph_count": 0,
            "structure_type": "simple",
        }

        heading_count = 0
        list_count = 0
        code_count = 0
        paragraph_count = 0

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue

            for pattern in StructureAnalyzer.ALL_HEADING_PATTERNS:
                if re.match(pattern, line_stripped, re.IGNORECASE):
                    analysis["has_headings"] = True
                    heading_count += 1
                    break

            for pattern in StructureAnalyzer.LIST_PATTERNS:
                if re.match(pattern, line_stripped):
                    analysis["has_lists"] = True
                    list_count += 1
                    break

            if "|" in line and line.count("|") >= 2:
                analysis["has_tables"] = True

            if line_stripped.startswith("```") or line_stripped.startswith(
                ("def ", "class ", "import ", "function ", "var ", "const ", "let ")
            ):
                analysis["has_code"] = True
                code_count += 1

            if len(line_stripped) > 10:
                paragraph_count += 1

        analysis["paragraph_count"] = paragraph_count
        if analysis["has_code"]:
            analysis["structure_type"] = "code"
        elif heading_count > paragraph_count * 0.2:
            analysis["structure_type"] = "hierarchical"
        elif analysis["has_tables"]:
            analysis["structure_type"] = "tabular"
        elif len(text) > 5000 and paragraph_count > 10:
            analysis["structure_type"] = "complex"
        else:
            analysis["structure_type"] = "simple"

        return analysis

    @staticmethod
    def extract_hierarchical_structure(text: str) -> List[Dict[str, Any]]:
        """Extract hierarchical structure."""
        lines = text.split("\n")
        hierarchy = []
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue
            for j, pattern in enumerate(StructureAnalyzer.ALL_HEADING_PATTERNS):
                if re.match(pattern, line_stripped, re.IGNORECASE):
                    hierarchy.append(
                        {"line_number": i, "level": (j % 4) + 1, "text": line_stripped}
                    )
                    break
        return hierarchy


class ChunkCache:
    """Simple caching mechanism for chunks."""

    def __init__(self, max_size: int = 1000):
        self._cache = {}
        self._access_order = []
        self.max_size = max_size

    def _make_key(self, text: str, config: ChunkingConfig) -> str:
        """Generate cache key."""
        content = (
            f"{text}_{config.chunk_size}_{config.chunk_overlap}_{config.min_chunk_size}"
        )
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, text: str, config: ChunkingConfig) -> Optional[List[Document]]:
        """Get cached result."""
        key = self._make_key(text, config)
        if key in self._cache:
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None

    def put(self, text: str, config: ChunkingConfig, result: List[Document]):
        """Cache result."""
        key = self._make_key(text, config)
        if len(self._cache) >= self.max_size and key not in self._cache:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]
        self._cache[key] = result
        if key not in self._access_order:
            self._access_order.append(key)


class Chunking:
    """Chunking with adaptive strategy and caching."""

    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config if config else ChunkingConfig()
        self.language_detector = LanguageDetector()
        self.structure_analyzer = StructureAnalyzer()
        self.cache = ChunkCache() if self.config.enable_caching else None
        self._init_splitters()

    def _init_splitters(self):
        """Initialize text splitters."""
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
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=self._count_tokens,
            is_separator_regex=False,
        )

    def _count_tokens(self, text: str) -> int:
        """Estimate token count."""
        if not text:
            return 0
        lang_info = self.language_detector.detect_language(text)
        if lang_info["language"] == "vietnamese":
            return len(text) // 3
        elif lang_info["language"] == "english":
            return len(text) // 4
        return int(len(text) / 3.5)

    def _prepare_chunk_metadata(
        self, chunk_text: str, base_metadata: Dict[str, Any], strategy: str, **kwargs
    ) -> Dict[str, Any]:
        """Prepare metadata for chunk."""
        lang_info = self.language_detector.detect_language(chunk_text)
        chunk_metadata = base_metadata.copy()
        chunk_metadata.update(
            {
                "strategy": strategy,
                "token_count": self._count_tokens(chunk_text),
                "language_info": lang_info,
                **kwargs,
            }
        )
        return chunk_metadata

    async def _chunk_simple(
        self, text: str, metadata: Dict[str, Any]
    ) -> List[Document]:
        """Simple chunking strategy."""
        if not text:
            return []
        chunks = self.splitter.split_text(text)
        documents = []
        doc_index = 0
        for chunk in chunks:
            chunk_stripped = chunk.strip()
            if not chunk_stripped:
                continue
            if (
                len(chunk_stripped) < self.config.min_chunk_size
                and documents
                and len(documents[-1].page_content) + len(chunk_stripped)
                < self.config.max_chunk_size
            ):
                documents[-1].page_content += "\n" + chunk_stripped
                documents[-1].metadata = self._prepare_chunk_metadata(
                    documents[-1].page_content,
                    metadata,
                    "simple",
                    chunk_index=doc_index - 1,
                )
                continue
            elif (
                len(chunk_stripped) < self.config.min_chunk_size
                and not self.config.keep_small_chunks
            ):
                continue
            chunk_metadata = self._prepare_chunk_metadata(
                chunk_stripped, metadata, "simple", chunk_index=doc_index
            )
            documents.append(
                Document(page_content=chunk_stripped, metadata=chunk_metadata)
            )
            doc_index += 1
        return documents

    async def _chunk_hierarchical(
        self, text: str, metadata: Dict[str, Any]
    ) -> List[Document]:
        """Hierarchical chunking strategy."""
        structure = self.structure_analyzer.extract_hierarchical_structure(text)
        if not structure:
            return await self._chunk_simple(text, metadata)
        lines = text.split("\n")
        sections = []
        current_section = []
        current_level = float("inf")
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                if current_section:
                    current_section.append(line)
                continue
            line_level = None
            for heading in structure:
                if heading["line_number"] == i:
                    line_level = heading["level"]
                    break
            if line_level is not None and line_level <= current_level:
                if current_section:
                    section_content = "\n".join(current_section)
                    sections.append(
                        {
                            "content": section_content,
                            "level": current_level,
                            "token_count": self._count_tokens(section_content),
                        }
                    )
                current_section = [line]
                current_level = line_level
            else:
                current_section.append(line)
        if current_section:
            section_content = "\n".join(current_section)
            sections.append(
                {
                    "content": section_content,
                    "level": current_level,
                    "token_count": self._count_tokens(section_content),
                }
            )
        documents = []
        for i, section in enumerate(sections):
            if section["token_count"] > self.config.max_chunk_size:
                sub_chunks = await self._chunk_simple(section["content"], metadata)
                for chunk in sub_chunks:
                    chunk.metadata = self._prepare_chunk_metadata(
                        chunk.page_content,
                        metadata,
                        "hierarchical_split",
                        section_level=section["level"],
                        chunk_index=i,
                    )
                documents.extend(sub_chunks)
            else:
                chunk_metadata = self._prepare_chunk_metadata(
                    section["content"],
                    metadata,
                    "hierarchical",
                    section_level=section["level"],
                    chunk_index=i,
                )
                documents.append(
                    Document(
                        page_content=section["content"].strip(), metadata=chunk_metadata
                    )
                )
        return documents

    async def _chunk_semantic(
        self, text: str, metadata: Dict[str, Any]
    ) -> List[Document]:
        """Semantic chunking strategy."""
        paragraphs = re.split(r"\n\s*\n", text)
        chunks = []
        current_chunk = []
        current_tokens = 0
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            para_tokens = self._count_tokens(paragraph)
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
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
        documents = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = self._prepare_chunk_metadata(
                chunk, metadata, "semantic", chunk_index=i
            )
            documents.append(Document(page_content=chunk, metadata=chunk_metadata))
        return documents

    async def _chunk_code(self, text: str, metadata: Dict[str, Any]) -> List[Document]:
        """Code chunking strategy."""
        code_separators = [
            "\n\nclass ",
            "\n\ndef ",
            "\n\nfunction ",
            "\n\n```",
            "\n\n",
            "\n",
            " ",
            "",
        ]
        code_splitter = RecursiveCharacterTextSplitter(
            separators=code_separators,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap // 2,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = code_splitter.split_text(text)
        documents = []
        for i, chunk in enumerate(chunks):
            chunk_stripped = chunk.strip()
            if not chunk_stripped:
                continue
            chunk_metadata = self._prepare_chunk_metadata(
                chunk_stripped, metadata, "code", chunk_index=i
            )
            documents.append(
                Document(page_content=chunk_stripped, metadata=chunk_metadata)
            )
        return documents

    async def chunk_text(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Chunk text with optimal strategy."""
        if not text or not text.strip():
            return []
        if self.cache:
            cached_result = self.cache.get(text, self.config)
            if cached_result:
                logger.debug("Using cached chunking result")
                return cached_result
        analysis = self.structure_analyzer.analyze_structure(text)
        chunk_metadata = (metadata or {}).copy()
        chunk_metadata.update(
            {"original_length": len(text), "structure_analysis": analysis}
        )
        structure_type = analysis["structure_type"]
        logger.info(f"Structure analysis: {structure_type}")
        if structure_type == "code":
            chunks = await self._chunk_code(text, chunk_metadata)
            logger.info(f"Using code chunking → {len(chunks)} chunks")
        elif structure_type == "hierarchical":
            chunks = await self._chunk_hierarchical(text, chunk_metadata)
            logger.info(f"Using hierarchical chunking → {len(chunks)} chunks")
        elif structure_type in ["complex", "tabular"]:
            chunks = await self._chunk_semantic(text, chunk_metadata)
            logger.info(f"Using semantic chunking → {len(chunks)} chunks")
        else:
            chunks = await self._chunk_simple(text, chunk_metadata)
            logger.info(f"Using simple chunking → {len(chunks)} chunks")
        if self.cache:
            self.cache.put(text, self.config, chunks)
        return chunks

    async def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Chunk multiple documents."""
        all_chunks = []
        for i, doc in enumerate(documents):
            logger.info(f"Processing document {i+1}/{len(documents)}")
            doc_metadata = doc.metadata.copy()
            doc_metadata["document_index"] = i
            chunks = await self.chunk_text(doc.page_content, doc_metadata)
            all_chunks.extend(chunks)
        logger.info(f"Total: {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks


_default_chunking = None
_cached_chunking = None


def get_chunking_instance(enable_caching: bool = True) -> Chunking:
    """Get singleton chunking instance."""
    global _default_chunking, _cached_chunking
    if enable_caching:
        if _cached_chunking is None:
            config = ChunkingConfig(enable_caching=True)
            _cached_chunking = Chunking(config)
        return _cached_chunking
    else:
        if _default_chunking is None:
            config = ChunkingConfig(enable_caching=False)
            _default_chunking = Chunking(config)
        return _default_chunking


async def chunk_documents_optimal(
    documents: List[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    progress_callback: Optional[Callable[[str, float], Awaitable[None]]] = None,
) -> List[Document]:
    """Chunk multiple documents optimally with async processing."""
    chunking = get_chunking_instance(enable_caching=True)
    chunking.config.chunk_size = chunk_size
    chunking.config.chunk_overlap = chunk_overlap
    total_docs = len(documents)
    batch_size = min(50, max(10, total_docs // (os.cpu_count() or 4)))
    all_chunks = []
    start_time = time.time()
    logger.info(f"Starting chunking {total_docs} documents (batch_size={batch_size})")
    for i in range(0, total_docs, batch_size):
        batch = documents[i : i + batch_size]
        batch_start_time = time.time()
        try:
            tasks = [
                chunking.chunk_text(doc.page_content, doc.metadata) for doc in batch
            ]
            batch_chunks = [
                chunk for task in await asyncio.gather(*tasks) for chunk in task
            ]
            all_chunks.extend(batch_chunks)
            completed = i + len(batch)
            progress = (completed / total_docs) * 100
            batch_time = time.time() - batch_start_time
            if progress_callback:
                await progress_callback(
                    f"Chunked {completed}/{total_docs} documents", progress
                )
            logger.info(
                f"Batch {(i//batch_size)+1}: {len(batch)} docs → {len(batch_chunks)} chunks "
                f"({batch_time:.2f}s) | Progress: {progress:.1f}%"
            )
            await asyncio.sleep(min(0.1, batch_time * 0.05))
        except Exception as e:
            logger.error(f"Error in batch {(i//batch_size)+1}: {e}")
            continue
    total_time = time.time() - start_time
    logger.info(
        f"Chunking completed: {len(all_chunks)} chunks from {total_docs} documents "
        f"in {total_time:.2f}s (avg: {total_time/total_docs:.3f}s/doc)"
    )
    return all_chunks


def get_chunking_stats(chunks: List[Document]) -> Dict[str, Any]:
    """Get statistics about chunks."""
    if not chunks:
        return {}
    token_counts = [chunk.metadata.get("token_count", 0) for chunk in chunks]
    strategies = [chunk.metadata.get("strategy", "unknown") for chunk in chunks]
    languages = [
        chunk.metadata.get("language_info", {}).get("language", "unknown")
        for chunk in chunks
    ]
    return {
        "total_chunks": len(chunks),
        "total_tokens": sum(token_counts),
        "avg_tokens_per_chunk": sum(token_counts) / len(chunks) if token_counts else 0,
        "min_tokens": min(token_counts) if token_counts else 0,
        "max_tokens": max(token_counts) if token_counts else 0,
        "strategies_used": dict(Counter(strategies)),
        "languages_detected": dict(Counter(languages)),
    }


async def main():
    """Main function to test chunking."""
    # Sample text (mixed Vietnamese and English)
    sample_text = """
    # Giới thiệu về Chunking

    Chunking là quá trình chia nhỏ văn bản thành các đoạn (chunks) để xử lý dễ dàng hơn.
    This process is crucial for natural language processing tasks, especially in multilingual contexts.

    ## Các phương pháp Chunking

    1. Simple Chunking: Chia văn bản dựa trên ký tự hoặc token.
    2. Hierarchical Chunking: Dựa trên cấu trúc tiêu đề (headings).
    - Semantic Chunking: Dựa trên ý nghĩa của đoạn văn.
    - Code Chunking: Dành cho mã nguồn.

    ```python
    def example_function():
        print("This is a code chunk")
    ```

    Trong tiếng Việt, chunking cần xử lý đặc thù ngôn ngữ như từ ghép.
    For example, "máy bay" should be treated as a single token.
    """

    # Create a sample document
    doc = Document(
        page_content=sample_text, metadata={"source": "test_document", "id": "001"}
    )

    # Initialize chunking
    chunking = get_chunking_instance(enable_caching=True)

    # Progress callback for demonstration
    async def progress_callback(message: str, progress: float):
        logger.info(f"Progress: {message} ({progress:.1f}%)")

    # Chunk documents
    chunks = await chunk_documents_optimal(
        documents=[doc],
        chunk_size=256,
        chunk_overlap=32,
        progress_callback=progress_callback,
    )

    # Print results
    logger.info("\n=== Chunking Results ===")
    for i, chunk in enumerate(chunks):
        logger.info(f"Chunk {i+1}:")
        logger.info(f"Content: {chunk.page_content[:100]}...")
        logger.info(f"Metadata: {chunk.metadata}")
        logger.info("-" * 50)

    # Print statistics
    stats = get_chunking_stats(chunks)
    logger.info("\n=== Chunking Statistics ===")
    logger.info(f"Stats: {stats}")


if __name__ == "__main__":
    asyncio.run(main())
