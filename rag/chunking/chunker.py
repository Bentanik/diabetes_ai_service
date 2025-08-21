import asyncio
from typing import List
import re
import logging
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

from rag.dataclasses import ChunkMetadata, DocumentChunk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class Chunker:
    """
    Semantic chunker that preserves sentence boundaries and meaning.
    Optimized for embedding and retrieval tasks.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1024,
        preserve_paragraphs: bool = True
    ):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Target number of words for each chunk
            chunk_overlap: Number of words to overlap between chunks
            min_chunk_size: Minimum number of words for a chunk
            max_chunk_size: Maximum number of words for a chunk
            preserve_paragraphs: Try to keep paragraphs together when possible
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.preserve_paragraphs = preserve_paragraphs
        self.logger = logging.getLogger(__name__)
        
    async def chunk_async(self, text: str) -> List[DocumentChunk]:
        return await asyncio.to_thread(self.chunk_text, text)
    
    def chunk_text(self, text: str) -> List[DocumentChunk]:
        """
        Split text into semantic chunks.
        
        Args:
            text: The text to chunk
            
        Returns:
            List of DocumentChunk objects
        """
        if not text or not text.strip():
            return []
        
        # Split by pages if present (PDF page separator)
        pages = text.split('\f')
        
        all_chunks = []
        
        for page_text in pages:
            if not page_text.strip():
                continue
                
            # Process each page
            page_chunks = self._chunk_page(page_text, len(all_chunks))
            all_chunks.extend(page_chunks)
            
        return all_chunks
    
    def _chunk_page(self, text: str, chunk_index_start: int) -> List[DocumentChunk]:
        """Chunk a single page of text"""
        
        # Split into paragraphs if preserve_paragraphs is True
        if self.preserve_paragraphs:
            paragraphs = self._split_paragraphs(text)
        else:
            paragraphs = [text]
        
        chunks = []
        current_chunk_text = ""
        current_word_count = 0
        chunk_index = chunk_index_start
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
                
            # Split paragraph into sentences
            sentences = self._split_sentences(paragraph)
            
            for sent_idx, sentence in enumerate(sentences):
                if not sentence.strip():
                    continue
                
                sentence_words = word_tokenize(sentence)
                sentence_word_count = len(sentence_words)
                
                # Check if adding this sentence would exceed chunk size
                if current_word_count + sentence_word_count > self.chunk_size and current_chunk_text:
                    # Save current chunk
                    chunk = self._create_chunk(
                        content=current_chunk_text.strip(),
                        chunk_index=chunk_index,
                        word_count=current_word_count
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    
                    # Start new chunk with overlap
                    overlap_text, overlap_word_count = self._get_overlap_text(
                        sentences[:sent_idx], 
                        current_chunk_text
                    )
                    
                    current_chunk_text = overlap_text + (" " if overlap_text else "") + sentence
                    current_word_count = overlap_word_count + sentence_word_count
                else:
                    # Add sentence to current chunk
                    if current_chunk_text:
                        current_chunk_text += " " + sentence
                    else:
                        current_chunk_text = sentence
                    current_word_count += sentence_word_count
                
                # Handle maximum chunk size
                if current_word_count > self.max_chunk_size:
                    # Split the current chunk if it's too long
                    sub_chunks = self._split_long_chunk(
                        current_chunk_text,
                        chunk_index
                    )
                    chunks.extend(sub_chunks)
                    chunk_index += len(sub_chunks)
                    current_chunk_text = ""
                    current_word_count = 0
                    
            # Add paragraph separator if we're preserving paragraphs
            if self.preserve_paragraphs and current_chunk_text:
                current_chunk_text += "\n\n"
                    
        # Don't forget the last chunk
        if current_chunk_text.strip() and current_word_count >= self.min_chunk_size:
            chunk = self._create_chunk(
                content=current_chunk_text.strip(),
                chunk_index=chunk_index,
                word_count=current_word_count
            )
            chunks.append(chunk)
            
        return chunks
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        # Split by double newlines or more
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences with improved handling"""
        # Use NLTK sentence tokenizer
        sentences = sent_tokenize(text)
        
        # Post-process sentences to handle common edge cases
        processed_sentences = []
        
        for i, sent in enumerate(sentences):
            sent = sent.strip()
            if not sent:
                continue
                
            # Check if this sentence should be merged with the previous one
            if processed_sentences and self._should_merge_sentences(
                processed_sentences[-1], sent
            ):
                processed_sentences[-1] = processed_sentences[-1] + " " + sent
            else:
                processed_sentences.append(sent)
                
        return processed_sentences
    
    def _should_merge_sentences(self, sent1: str, sent2: str) -> bool:
        """Check if two sentences should be merged"""
        # Common abbreviations that don't end sentences
        abbreviations = [
            'Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Prof.', 'Sr.', 'Jr.', 
            'Ph.D', 'M.D', 'B.A', 'M.A', 'B.S', 'M.S',
            'i.e.', 'e.g.', 'etc.', 'vs.', 'Inc.', 'Ltd.', 'Co.'
        ]
        
        # Check if first sentence ends with an abbreviation
        for abbr in abbreviations:
            if sent1.endswith(abbr):
                return True
        
        # Check if second sentence starts with lowercase (continuation)
        if sent2 and sent2[0].islower():
            return True
        
        # Check for very short sentences that might be fragments
        if len(sent1.split()) < 3 or len(sent2.split()) < 3:
            # But don't merge if they look like complete sentences
            if not (sent1.endswith('.') and sent2[0].isupper()):
                return True
                
        return False
    
    def _get_overlap_text(
        self, 
        previous_sentences: List[str], 
        current_chunk: str
    ) -> tuple[str, int]:
        """Get overlap text from the end of current chunk"""
        if not previous_sentences or not self.chunk_overlap:
            return "", 0
            
        # Get sentences from the end of current chunk
        chunk_sentences = self._split_sentences(current_chunk)
        if not chunk_sentences:
            return "", 0
            
        overlap_sentences = []
        overlap_word_count = 0
        
        # Go backwards and collect sentences until we reach overlap size
        for sent in reversed(chunk_sentences):
            sent_word_count = len(word_tokenize(sent))
            
            if overlap_word_count + sent_word_count > self.chunk_overlap:
                break
                
            overlap_sentences.insert(0, sent)
            overlap_word_count += sent_word_count
            
        return " ".join(overlap_sentences), overlap_word_count
    
    def _split_long_chunk(self, text: str, chunk_index: int) -> List[DocumentChunk]:
        """Split a chunk that's too long into smaller chunks"""
        chunks = []
        sentences = self._split_sentences(text)
        
        current_chunk = ""
        current_word_count = 0
        
        for sentence in sentences:
            sentence_words = word_tokenize(sentence)
            sentence_word_count = len(sentence_words)
            
            if sentence_word_count > self.max_chunk_size:
                # Handle very long sentences by splitting at punctuation
                sub_sentences = self._split_long_sentence(sentence)
                
                for sub_sent in sub_sentences:
                    sub_word_count = len(word_tokenize(sub_sent))
                    
                    if current_word_count + sub_word_count > self.max_chunk_size:
                        if current_chunk:
                            chunks.append(self._create_chunk(
                                current_chunk.strip(),
                                chunk_index + len(chunks),
                                current_word_count
                            ))
                            current_chunk = sub_sent
                            current_word_count = sub_word_count
                    else:
                        current_chunk += " " + sub_sent if current_chunk else sub_sent
                        current_word_count += sub_word_count
            else:
                if current_word_count + sentence_word_count > self.max_chunk_size:
                    if current_chunk:
                        chunks.append(self._create_chunk(
                            current_chunk.strip(),
                            chunk_index + len(chunks),
                            current_word_count
                        ))
                        current_chunk = sentence
                        current_word_count = sentence_word_count
                else:
                    current_chunk += " " + sentence if current_chunk else sentence
                    current_word_count += sentence_word_count
        
        if current_chunk:
            chunks.append(self._create_chunk(
                current_chunk.strip(),
                chunk_index + len(chunks),
                current_word_count
            ))
            
        return chunks
    
    def _split_long_sentence(self, sentence: str) -> List[str]:
        """Split a very long sentence at natural breaking points"""
        # Try to split at punctuation marks that often indicate clauses
        split_patterns = [
            (r'; ', '; '),
            (r', ', ', '),
            (r' - ', ' - '),
            (r' — ', ' — '),
            (r' \(', ' ('),
            (r'\) ', ') '),
        ]
        
        parts = [sentence]
        
        for pattern, separator in split_patterns:
            new_parts = []
            for part in parts:
                if len(word_tokenize(part)) > self.max_chunk_size:
                    # Split and keep the separator
                    splits = re.split(f'({pattern})', part)
                    # Reconstruct with separators
                    temp = ""
                    for split in splits:
                        if split.strip():
                            if re.match(pattern, split):
                                temp += split
                            else:
                                if temp:
                                    new_parts.append(temp)
                                temp = split
                    if temp:
                        new_parts.append(temp)
                else:
                    new_parts.append(part)
            parts = new_parts
            
        return parts
    
    def _create_chunk(
        self,
        content: str,
        chunk_index: int,
        word_count: int
    ) -> DocumentChunk:
        """Create a DocumentChunk object"""
        metadata = ChunkMetadata(
            chunk_index=chunk_index,
            word_count=word_count,
            chunking_strategy="semantic"
        )
        
        return DocumentChunk(
            content=content,
            metadata=metadata
        )


# Example usage
if __name__ == "__main__":
    # Create chunker instance
    chunker = Chunker(
        chunk_size=512,      # Target 512 words per chunk
        chunk_overlap=128,   # 128 words overlap
        min_chunk_size=100,  # Minimum 100 words
        max_chunk_size=1024, # Maximum 1024 words
        preserve_paragraphs=True
    )
    
    # Example text
    sample_text = """
    This is the first paragraph of the document. It contains multiple sentences. 
    Each sentence should be kept together when possible.
    
    This is the second paragraph. Dr. Smith and Mr. Johnson discussed the project.
    The meeting was productive. They decided to proceed with the implementation.
    
    The third paragraph contains technical details about the system architecture.
    It includes various components and their interactions.
    """
    
    # Chunk the text
    chunks = chunker.chunk_text(sample_text)
    # Print results
    for chunk in chunks:
        print(f"Chunk {chunk.metadata.chunk_index}:")
        print(f"  Words: {chunk.metadata.word_count}")
        print(f"  Strategy: {chunk.metadata.chunking_strategy}")
        print(f"  Content: {chunk.content[:100]}...")
        print()
