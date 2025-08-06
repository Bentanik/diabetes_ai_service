import asyncio
from concurrent.futures import ThreadPoolExecutor
import re
import logging
import hashlib
from typing import List, Optional, TypedDict
from dataclasses import dataclass
from functools import lru_cache

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import nltk
from nltk.corpus import stopwords

try:
    from underthesea import word_tokenize as vi_tokenize

    HAS_VIETNAMESE_TOKENIZER = True
except ImportError:
    HAS_VIETNAMESE_TOKENIZER = False
    vi_tokenize = None

logger = logging.getLogger(__name__)


# Type definitions
class LanguageInfo(TypedDict):
    language: str
    vietnamese_ratio: float
    confidence: float


class StructureAnalysis(TypedDict):
    has_headings: bool
    has_lists: bool
    has_tables: bool
    has_code: bool
    paragraph_count: int
    structure_type: str


class HierarchicalStructure(TypedDict):
    line_number: int
    level: int
    text: str


class ChunkMetadata(TypedDict):
    source: Optional[str]
    id: Optional[str]
    author: Optional[str]
    created_at: Optional[str]
    original_length: Optional[int]
    structure_analysis: Optional[StructureAnalysis]
    strategy: str
    token_count: int
    language_info: LanguageInfo
    processed_by_llm: bool
    section_level: Optional[float]
    chunk_index: Optional[int]


class Chunk(TypedDict):
    text: str
    metadata: ChunkMetadata


@dataclass
class ChunkingConfig:
    """Cấu hình cho chunking"""

    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_size: int = 50
    max_chunk_size: int = 1024
    keep_small_chunks: bool = True
    enable_caching: bool = True


class MultilingualTextProcessor:
    """Text processor cho văn bản đa ngôn ngữ"""

    ALL_HEADING_PATTERNS = [
        r"^(?:CHƯƠNG|PHẦN|MỤC|BÀI|TIẾT|PHỤLỤC|PHỤ LỤC|CHAPTER|SECTION|PART|ARTICLE|APPENDIX)\s+[IVXLCDM\d]+",
        r"^(?:\d+|[IVXLCDM]+)\.\s+[A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐĐ]",
        r"^#{1,6}\s+",
        r"^[A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐĐ][A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐĐ\s]{2,}$",
    ]
    LIST_PATTERNS = [
        r"^(?:[-•*]|\d+[\.\)]|[a-zA-Z][\.\)]|[IVXLCDM]+[\.\)])\s+",
    ]
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
        "thì",
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
    _english_stopwords: Optional[set] = None
    _nltk_initialized: bool = False

    @classmethod
    async def _ensure_nltk_data(cls) -> None:
        if cls._nltk_initialized:
            return
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            logger.info("Downloading NLTK stopwords...")
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: nltk.download("stopwords", quiet=True)
            )
        cls._english_stopwords = set(stopwords.words("english"))
        cls._nltk_initialized = True

    @staticmethod
    @lru_cache(maxsize=1000)
    def detect_language(text: str) -> LanguageInfo:
        if not text:
            return {"language": "unknown", "vietnamese_ratio": 0.0, "confidence": 0.0}
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
            return {"language": "unknown", "vietnamese_ratio": 0.0, "confidence": 0.0}
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

    async def analyze_structure(self, text: str) -> StructureAnalysis:
        if not text:
            return {
                "has_headings": False,
                "has_lists": False,
                "has_tables": False,
                "has_code": False,
                "paragraph_count": 0,
                "structure_type": "simple",
            }
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._analyze_structure_sync, text)

    @staticmethod
    def _analyze_structure_sync(text: str) -> StructureAnalysis:
        lines = text.split("\n")
        analysis: StructureAnalysis = {
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
            for pattern in MultilingualTextProcessor.ALL_HEADING_PATTERNS:
                if re.match(pattern, line_stripped, re.IGNORECASE):
                    analysis["has_headings"] = True
                    heading_count += 1
                    break
            for pattern in MultilingualTextProcessor.LIST_PATTERNS:
                if re.match(pattern, line_stripped):
                    analysis["has_lists"] = True
                    list_count += 1
                    break
            if "|" in line and line.count("|") >= 2:
                analysis["has_tables"] = True
            if (
                line_stripped.startswith("```")
                or line_stripped.startswith("def ")
                or line_stripped.startswith("class ")
                or line_stripped.startswith("import ")
                or line_stripped.startswith("function ")
                or line_stripped.startswith("var ")
                or line_stripped.startswith("const ")
                or line_stripped.startswith("let ")
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

    async def extract_hierarchical_structure(
        self, text: str
    ) -> List[HierarchicalStructure]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._extract_hierarchical_structure_sync, text
        )

    @staticmethod
    def _extract_hierarchical_structure_sync(text: str) -> List[HierarchicalStructure]:
        lines = text.split("\n")
        hierarchy: List[HierarchicalStructure] = []
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue
            for j, pattern in enumerate(MultilingualTextProcessor.ALL_HEADING_PATTERNS):
                match = re.match(pattern, line_stripped, re.IGNORECASE)
                if match:
                    hierarchy.append(
                        {
                            "line_number": i,
                            "level": (j % 4) + 1,
                            "text": line_stripped,
                        }
                    )
                    break
        return hierarchy

    async def tokenize_text(self, text: str, lang_info: LanguageInfo) -> List[str]:
        await self._ensure_nltk_data()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._tokenize_text_sync, text, lang_info
        )

    def _tokenize_text_sync(self, text: str, lang_info: LanguageInfo) -> List[str]:
        if lang_info["language"] == "vietnamese" and HAS_VIETNAMESE_TOKENIZER:
            try:
                if vi_tokenize is not None:
                    tokens = vi_tokenize(text)
                else:
                    tokens = text.split()
                return [t for t in tokens if t.lower() not in self.VIETNAMESE_STOPWORDS]
            except Exception as e:
                logger.warning(
                    f"Vietnamese tokenization failed: {e}, falling back to simple tokenization"
                )
        tokens = re.findall(r"\b\w+\b", text.lower())
        return [t for t in tokens if t not in self._english_stopwords]


class ChunkCache:
    def __init__(self, max_size: int = 1000):
        self._cache: dict[str, List[Document]] = {}
        self._access_order: List[str] = []
        self.max_size = max_size

    def _make_key(self, text: str, config: ChunkingConfig) -> str:
        content = (
            f"{text}_{config.chunk_size}_{config.chunk_overlap}_{config.min_chunk_size}"
        )
        return hashlib.sha256(content.encode()).hexdigest()

    async def get(self, text: str, config: ChunkingConfig) -> Optional[List[Document]]:
        key = self._make_key(text, config)
        if key in self._cache:
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None

    async def put(
        self, text: str, config: ChunkingConfig, result: List[Document]
    ) -> None:
        key = self._make_key(text, config)
        if len(self._cache) >= self.max_size and key not in self._cache:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]
        self._cache[key] = result
        if key not in self._access_order:
            self._access_order.append(key)


class Chunking:
    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config if config is not None else ChunkingConfig()
        self.text_processor = MultilingualTextProcessor()
        self.cache: Optional[ChunkCache] = (
            ChunkCache() if self.config.enable_caching else None
        )
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._init_splitters()

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
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=self._count_tokens,
            is_separator_regex=False,
        )

    def _count_tokens(self, text: str) -> int:
        if not text:
            return 0
        lang_info = self.text_processor.detect_language(text)
        if lang_info["language"] == "vietnamese":
            return len(text) // 3
        elif lang_info["language"] == "english":
            return len(text) // 4
        else:
            return int(len(text) / 3.5)

    async def _prepare_chunk_metadata(
        self,
        chunk_text: str,
        base_metadata: Optional[ChunkMetadata],
        strategy: str,
        **kwargs,
    ) -> ChunkMetadata:
        lang_info = self.text_processor.detect_language(chunk_text)
        chunk_metadata: ChunkMetadata = (base_metadata or {}).copy()
        chunk_metadata.update(
            {
                "strategy": strategy,
                "token_count": self._count_tokens(chunk_text),
                "language_info": lang_info,
                "processed_by_llm": False,
                **kwargs,
            }
        )
        return chunk_metadata

    async def _chunk_simple(
        self, text: str, metadata: Optional[ChunkMetadata]
    ) -> List[Document]:
        if not text:
            return []
        loop = asyncio.get_event_loop()
        chunks = await loop.run_in_executor(
            self.executor, self.splitter.split_text, text
        )
        documents: List[Document] = []
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
                documents[-1].metadata = await self._prepare_chunk_metadata(
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
            chunk_metadata = await self._prepare_chunk_metadata(
                chunk_stripped, metadata, "simple", chunk_index=doc_index
            )
            documents.append(
                Document(page_content=chunk_stripped, metadata=chunk_metadata)
            )
            doc_index += 1
        return documents

    async def _chunk_hierarchical(
        self, text: str, metadata: Optional[ChunkMetadata]
    ) -> List[Document]:
        structure = await self.text_processor.extract_hierarchical_structure(text)
        if not structure:
            return await self._chunk_simple(text, metadata)
        lines = text.split("\n")
        sections: List[dict] = []
        current_section: List[str] = []
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
        documents: List[Document] = []
        for i, section in enumerate(sections):
            if section["token_count"] > self.config.max_chunk_size:
                sub_chunks = await self._chunk_simple(section["content"], metadata)
                for chunk in sub_chunks:
                    chunk.metadata = await self._prepare_chunk_metadata(
                        chunk.page_content,
                        metadata,
                        "hierarchical_split",
                        section_level=section["level"],
                        chunk_index=i,
                    )
                documents.extend(sub_chunks)
            else:
                chunk_metadata = await self._prepare_chunk_metadata(
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
        self, text: str, metadata: Optional[ChunkMetadata]
    ) -> List[Document]:
        paragraphs = re.split(r"\n\s*\n", text)
        chunks: List[str] = []
        current_chunk: List[str] = []
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
            chunk_content = "\n\n".join(current_chunk)
            chunks.append(chunk_content)
        documents: List[Document] = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = await self._prepare_chunk_metadata(
                chunk, metadata, "semantic", chunk_index=i
            )
            documents.append(Document(page_content=chunk, metadata=chunk_metadata))
        return documents

    async def _chunk_code(
        self, text: str, metadata: Optional[ChunkMetadata]
    ) -> List[Document]:
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
        loop = asyncio.get_event_loop()
        chunks = await loop.run_in_executor(
            self.executor, code_splitter.split_text, text
        )
        documents: List[Document] = []
        for i, chunk in enumerate(chunks):
            chunk_stripped = chunk.strip()
            if not chunk_stripped:
                continue
            chunk_metadata = await self._prepare_chunk_metadata(
                chunk_stripped, metadata, "code", chunk_index=i
            )
            documents.append(
                Document(page_content=chunk_stripped, metadata=chunk_metadata)
            )
        return documents

    async def chunk_text(
        self, text: str, metadata: Optional[ChunkMetadata] = None
    ) -> List[Chunk]:
        """Chunk text với strategy tối ưu, trả về danh sách Chunk"""
        if not text or not text.strip():
            return []
        # Check cache first
        if self.cache:
            cached_result = await self.cache.get(text, self.config)
            if cached_result:
                logger.debug("Using cached chunking result")
                return [
                    {"text": chunk.page_content, "metadata": chunk.metadata}
                    for chunk in cached_result
                ]
        # Prepare metadata
        chunk_metadata: ChunkMetadata = (metadata or {}).copy()
        # Check text length
        if len(text.strip()) < self.config.min_chunk_size:
            chunk_metadata.update(
                {
                    "original_length": len(text),
                    "structure_analysis": {
                        "has_headings": False,
                        "has_lists": False,
                        "has_tables": False,
                        "has_code": False,
                        "paragraph_count": 0,
                        "structure_type": "short",
                    },
                }
            )
            chunk_metadata = await self._prepare_chunk_metadata(
                text.strip(), chunk_metadata, "short"
            )
            result: List[Chunk] = [{"text": text.strip(), "metadata": chunk_metadata}]
            if self.cache:
                await self.cache.put(
                    text,
                    self.config,
                    [Document(page_content=text.strip(), metadata=chunk_metadata)],
                )
            return result
        # Analyze structure
        analysis = await self.text_processor.analyze_structure(text)
        chunk_metadata.update(
            {"original_length": len(text), "structure_analysis": analysis}
        )
        # Choose strategy
        structure_type = analysis["structure_type"]
        logger.info(f"Phân tích cấu trúc: {structure_type}")
        if structure_type == "code":
            chunks = await self._chunk_code(text, chunk_metadata)
            logger.info(f"Sử dụng chunking code → {len(chunks)} chunks")
        elif structure_type == "hierarchical":
            chunks = await self._chunk_hierarchical(text, chunk_metadata)
            logger.info(f"Sử dụng chunking phân cấp → {len(chunks)} chunks")
        elif structure_type in ["complex", "tabular"]:
            chunks = await self._chunk_semantic(text, chunk_metadata)
            logger.info(f"Sử dụng chunking ngữ nghĩa → {len(chunks)} chunks")
        else:
            chunks = await self._chunk_simple(text, chunk_metadata)
            logger.info(f"Sử dụng chunking đơn giản → {len(chunks)} chunks")
        # Convert to output format
        result: List[Chunk] = [
            {"text": chunk.page_content, "metadata": chunk.metadata} for chunk in chunks
        ]
        # Cache result
        if self.cache:
            await self.cache.put(text, self.config, chunks)
        return result


# Global instances for performance
_default_chunking: Optional[Chunking] = None
_cached_chunking: Optional[Chunking] = None


async def get_chunking_instance(enable_caching: bool = True) -> Chunking:
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


async def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    metadata: Optional[ChunkMetadata] = None,
    enable_caching: bool = True,
) -> List[Chunk]:
    """
    Chunk text với cấu hình tối ưu, trả về danh sách Chunk

    Args:
        text: Text cần chunk
        chunk_size: Kích thước chunk mong muốn
        chunk_overlap: Overlap between chunks
        metadata: Optional metadata từ bên ngoài
        enable_caching: Enable caching for performance

    Returns:
        List of Chunk dictionaries containing chunk text and metadata
    """
    config = ChunkingConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        enable_caching=enable_caching,
    )
    chunking = await get_chunking_instance(enable_caching)
    return await chunking.chunk_text(text, metadata)


async def analyze_text_structure(text: str) -> StructureAnalysis:
    processor = MultilingualTextProcessor()
    return await processor.analyze_structure(text)


async def detect_text_language(text: str) -> LanguageInfo:
    return MultilingualTextProcessor.detect_language(text)
