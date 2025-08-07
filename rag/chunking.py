import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional
import threading
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from rag.schemas.chunk import Chunk, PDFChunkMetadata
from rag.schemas.common.language_info import LanguageType
from rag.schemas.pdf import TextBlock
from rag.config import ChunkingConfig


def ensure_language_info(language_info):
    if not language_info:
        return None
    if isinstance(language_info, dict):
        class LangInfo:
            def __init__(self, d):
                self.language = d.get("language", LanguageType.UNKNOWN)
                self.confidence = d.get("confidence", 0.0)
        return LangInfo(language_info)
    return language_info


class ThreadSafeTokenizer:
    """Thread-safe wrapper cho tokenizer"""
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._local = threading.local()
    
    def _get_tokenizer(self):
        if not hasattr(self._local, 'tokenizer'):
            self._local.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._local.tokenizer
    
    def encode(self, text: str, **kwargs):
        return self._get_tokenizer().encode(text, **kwargs)
    
    def decode(self, tokens, **kwargs):
        return self._get_tokenizer().decode(tokens, **kwargs)


class Chunking:
    def __init__(
        self,
        config: Optional[ChunkingConfig] = None,
        model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
    ):
        self.logger = logging.getLogger(__name__)
        self.config = config if config else ChunkingConfig()
        self.model_name = model_name
        
        self.tokenizer = ThreadSafeTokenizer(model_name)

        if self.config.max_chunk_size > 512:
            self.logger.warning(f"max_chunk_size ({self.config.max_chunk_size}) > 512, setting to 400")
            self.config.max_chunk_size = 400
        
        if self.config.min_chunk_size > self.config.max_chunk_size // 2:
            self.config.min_chunk_size = self.config.max_chunk_size // 4

        self.executor = ThreadPoolExecutor(max_workers=2)
        self.semaphore = asyncio.Semaphore(2)

        self.separators_default = [
            "\n\n\n", "\n\n", "\n", ".", "!", "?", ";", ":", "...", ",", " ", ""
        ]

        self._init_splitter()

    def _init_splitter(self) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            separators=self.separators_default,
            chunk_size=self.config.max_chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=self._count_tokens,
            is_separator_regex=False,
        )

    def _count_tokens(self, text: str) -> int:
        try:
            tokens = self.tokenizer.encode(text, truncation=True, max_length=512, add_special_tokens=False)
            return len(tokens)
        except Exception as e:
            self.logger.warning(f"Token counting error: {e}, using word count")
            return int(len(text.split()) * 1.3)

    def _safe_truncate_by_chars(self, text: str, max_tokens: int = 400) -> str:
        if self._count_tokens(text) <= max_tokens:
            return text
        
        estimated_max_chars = max_tokens * 3
        if len(text) <= estimated_max_chars:
            return text
        
        truncated = text[:estimated_max_chars]
        for separator in ["\n\n", "\n", ".", "!", "?", " "]:
            last_sep = truncated.rfind(separator)
            if last_sep > estimated_max_chars * 0.8:
                return truncated[:last_sep + len(separator)]
        return truncated

    def _chunk_semantic_sync(self, block: TextBlock) -> List[Chunk]:
        text = block.context.strip()
        if not text:
            return []

        text = self._safe_truncate_by_chars(text, self.config.max_chunk_size)

        content_type = getattr(block.metadata, "block_type", "paragraph")
        if content_type == "paragraph" and any(text.startswith(s) for s in ["-", "*", "+"]):
            content_type = "list"

        token_count = self._count_tokens(text)

        if token_count <= self.config.min_chunk_size or content_type in ["heading", "code", "list"]:
            metadata = PDFChunkMetadata(block_metadata=block.metadata)
            return [Chunk(text=text, metadata=metadata)]

        lang_info = ensure_language_info(getattr(block.metadata, "language_info", None))
        language = lang_info.language if lang_info else LanguageType.UNKNOWN

        separators = self.separators_default
        if language == LanguageType.VIETNAMESE:
            separators = ["\n\n\n", "\n\n", "\n", ".", "!", "?", "...", ",", ";", ":"]

        splitter = RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=self.config.max_chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=self._count_tokens,
            is_separator_regex=False,
        )

        try:
            chunks = splitter.split_text(text)
        except Exception as e:
            self.logger.warning(f"Splitter error: {e}, using manual split")
            chunks = self._manual_split_preserve_format(text)

        if not chunks:
            chunks = [text]

        safe_chunks = []
        for chunk in chunks:
            safe_chunk = self._safe_truncate_by_chars(chunk, self.config.max_chunk_size)
            if safe_chunk.strip():
                safe_chunks.append(safe_chunk)
        
        chunks = safe_chunks

        merged_chunks = []
        current_text = ""
        current_tokens = 0
        
        for chunk in chunks:
            chunk_tokens = self._count_tokens(chunk)
            
            if current_text and current_tokens + chunk_tokens <= self.config.max_chunk_size:
                current_text += "\n\n" + chunk
                current_tokens += chunk_tokens
            else:
                if current_text and current_tokens >= self.config.min_chunk_size:
                    metadata = PDFChunkMetadata(block_metadata=block.metadata)
                    merged_chunks.append(Chunk(text=current_text, metadata=metadata))
                current_text = chunk
                current_tokens = chunk_tokens

        if current_text:
            if merged_chunks and current_tokens < self.config.min_chunk_size:
                last_chunk_text = merged_chunks[-1].text + "\n\n" + current_text
                if self._count_tokens(last_chunk_text) > self.config.max_chunk_size:
                    safe_merged = self._safe_truncate_by_chars(last_chunk_text, self.config.max_chunk_size)
                    merged_chunks[-1].text = safe_merged
                else:
                    merged_chunks[-1].text = last_chunk_text
            else:
                metadata = PDFChunkMetadata(block_metadata=block.metadata)
                merged_chunks.append(Chunk(text=current_text, metadata=metadata))

        return merged_chunks

    def _manual_split_preserve_format(self, text: str) -> List[str]:
        paragraphs = re.split(r'\n\s*\n', text)
        if len(paragraphs) > 1:
            chunks = []
            current_chunk = ""
            current_tokens = 0
            
            for para in paragraphs:
                para_tokens = self._count_tokens(para)
                if current_chunk and current_tokens + para_tokens <= self.config.max_chunk_size:
                    current_chunk += "\n\n" + para
                    current_tokens += para_tokens
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = para
                    current_tokens = para_tokens
            
            if current_chunk:
                chunks.append(current_chunk)
            return chunks
        
        sentences = re.split(r'([.!?]+\s*)', text)
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            if i + 1 < len(sentences) and re.match(r'[.!?]+\s*', sentences[i + 1]):
                sentence += sentences[i + 1]
                i += 2
            else:
                i += 1
                
            sentence_tokens = self._count_tokens(sentence)
            if current_chunk and current_tokens + sentence_tokens <= self.config.max_chunk_size:
                current_chunk += sentence
                current_tokens += sentence_tokens
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
                current_tokens = sentence_tokens
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks if chunks else [text]

    async def chunk_text(self, blocks: List[TextBlock]) -> List[Chunk]:
        all_chunks = []

        async def run_chunk(block: TextBlock) -> List[Chunk]:
            async with self.semaphore:
                loop = asyncio.get_running_loop()
                try:
                    return await loop.run_in_executor(self.executor, self._chunk_semantic_sync, block)
                except Exception as e:
                    self.logger.error(f"Chunk processing error for block: {e}")
                    return []

        batch_size = 5
        for i in range(0, len(blocks), batch_size):
            batch = blocks[i:i + batch_size]
            tasks = [asyncio.create_task(run_chunk(block)) for block in batch]
            
            for task in asyncio.as_completed(tasks):
                try:
                    chunks = await task
                    all_chunks.extend(chunks)
                except Exception as e:
                    self.logger.error(f"Task completion error: {e}")
            
            await asyncio.sleep(0.1)

        return all_chunks

    def close(self):
        self.executor.shutdown(wait=True)


async def main():
    from rag.document_parser.pdf_extractor import PdfExtractor
    logging.basicConfig(level=logging.INFO)

    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    pdf_path = "C:/Users/Quangdepzai/Downloads/text.pdf"

    chunking_config = ChunkingConfig()
    chunking_config.max_chunk_size = 400
    chunking_config.min_chunk_size = 50
    chunking_config.chunk_overlap = 50

    extractor = PdfExtractor(
        enable_text_cleaning=True,
        remove_urls=True,
        remove_page_numbers=True,
        remove_short_lines=True,
        remove_newlines=True,
        remove_references=True,
        remove_stopwords=False,
        min_line_length=3,
        max_block_length=300,
        max_bbox_distance=50.0,
        debug_mode=True,
    )

    try:
        pages_data = await extractor.extract_all_pages_data(pdf_path)
        blocks: List[TextBlock] = [block for page_data in pages_data for block in page_data.blocks]

        chunker = Chunking(config=chunking_config, model_name=model_name)
        try:
            chunks = await chunker.chunk_text(blocks)
        finally:
            chunker.close()

        import json
        from dataclasses import asdict
        with open("chunks.json", "w", encoding="utf-8") as f:
            json.dump([asdict(chunk) for chunk in chunks], f, ensure_ascii=False, indent=2)

    except Exception as e:
        logging.error(f"Main execution error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
