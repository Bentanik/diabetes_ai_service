import asyncio
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from typing import List, Optional, Tuple

class RerankingModel:
    _instance: Optional["RerankingModel"] = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __init__(self) -> None:
        self.model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.model: Optional[CrossEncoder] = None
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None
        self._is_loaded = False

    async def load(self) -> None:
        if self._is_loaded:
            return

        def _load():
            try:
                self.model = CrossEncoder(
                    self.model_name,
                    device="cuda",
                    trust_remote_code=True,
                )
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            except Exception as e:
                raise RuntimeError(f"Failed to load reranking model or tokenizer: {e}")

        await asyncio.to_thread(_load)
        self._is_loaded = True

    @classmethod
    async def get_instance(cls) -> "RerankingModel":
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
                await cls._instance.load()
        return cls._instance

    async def rerank(self, query: str, documents: List[str], top_k: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        Rerank documents based on relevance to query
        Returns: List of (document, score) tuples sorted by relevance score
        """
        if not self._is_loaded:
            raise RuntimeError("RerankingModel chưa được load. Gọi await load() trước.")
        
        if not documents:
            return []

        def _rerank():
            # Tạo pairs (query, document) cho cross-encoder
            pairs = [(query, doc) for doc in documents]
            scores = self.model.predict(pairs)
            
            # Kết hợp documents với scores và sắp xếp theo điểm số giảm dần
            doc_scores = list(zip(documents, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Trả về top_k nếu được chỉ định
            if top_k is not None:
                doc_scores = doc_scores[:top_k]
                
            return doc_scores

        return await asyncio.to_thread(_rerank)

    async def get_scores(self, query: str, documents: List[str]) -> List[float]:
        """
        Get relevance scores for documents without sorting
        Returns: List of scores in the same order as input documents
        """
        if not self._is_loaded:
            raise RuntimeError("RerankingModel chưa được load. Gọi await load() trước.")
        
        if not documents:
            return []

        def _get_scores():
            pairs = [(query, doc) for doc in documents]
            scores = self.model.predict(pairs)
            return scores.tolist() if hasattr(scores, 'tolist') else list(scores)

        return await asyncio.to_thread(_get_scores)

    async def rerank_with_metadata(self, query: str, doc_metadata_pairs: List[Tuple[str, dict]], 
                                 top_k: Optional[int] = None) -> List[Tuple[str, dict, float]]:
        """
        Rerank documents with metadata
        Args:
            query: Search query
            doc_metadata_pairs: List of (document, metadata) tuples
            top_k: Number of top results to return
        Returns: List of (document, metadata, score) tuples sorted by relevance
        """
        if not self._is_loaded:
            raise RuntimeError("RerankingModel chưa được load. Gọi await load() trước.")
        
        if not doc_metadata_pairs:
            return []

        def _rerank_with_metadata():
            documents = [doc for doc, _ in doc_metadata_pairs]
            metadata_list = [meta for _, meta in doc_metadata_pairs]
            
            pairs = [(query, doc) for doc in documents]
            scores = self.model.predict(pairs)
            
            # Kết hợp documents, metadata và scores
            results = list(zip(documents, metadata_list, scores))
            results.sort(key=lambda x: x[2], reverse=True)
            
            if top_k is not None:
                results = results[:top_k]
                
            return results

        return await asyncio.to_thread(_rerank_with_metadata)

    def count_tokens(self, text: str) -> int:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer chưa được load. Gọi await load() trước.")
        return len(self.tokenizer.encode(text))

    def get_max_length(self) -> int:
        """Get maximum sequence length supported by the model"""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer chưa được load. Gọi await load() trước.")
        return getattr(self.tokenizer, 'model_max_length', 512)

# Example usage
async def example_usage():
    # Khởi tạo reranking model
    reranker = await RerankingModel.get_instance()
    
    query = "What is machine learning?"
    documents = [
        "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
        "Python is a popular programming language for data science.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing helps computers understand human language."
    ]
    
    # Rerank documents
    reranked_results = await reranker.rerank(query, documents, top_k=3)
    
    print("Reranked results:")
    for i, (doc, score) in enumerate(reranked_results, 1):
        print(f"{i}. Score: {score:.4f}")
        print(f"   Document: {doc[:100]}...")
        print()

if __name__ == "__main__":
    asyncio.run(example_usage())
