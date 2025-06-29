"""
Hybrid Retriever kết hợp BM25 và Embedding Search.
Cải thiện chất lượng retrieval bằng cách kết hợp keyword và semantic search.
"""

import math
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter, defaultdict
from langchain.schema import Document
from langchain_core.vectorstores import VectorStore
from core.logging_config import get_logger

logger = get_logger(__name__)


class BM25Retriever:
    """BM25 (Best Matching 25) implementation cho keyword search."""

    def __init__(self, documents: List[Document], k1: float = 1.5, b: float = 0.75):
        """
        Khởi tạo BM25 retriever.

        Args:
            documents: Danh sách documents
            k1: Term frequency saturation point
            b: Length normalization parameter
        """
        self.documents = documents
        self.k1 = k1
        self.b = b

        # Build inverted index và tính toán statistics
        self._build_index()

        logger.info(f"Khởi tạo BM25 với {len(documents)} documents")

    def _build_index(self):
        """Xây dựng inverted index và tính statistics."""
        self.doc_frequencies = {}  # Term document frequency
        self.idf = {}  # Inverse document frequency
        self.doc_len = {}  # Document lengths
        self.avgdl = 0  # Average document length

        # Tokenize tất cả documents
        all_tokens = []
        for i, doc in enumerate(self.documents):
            tokens = self._tokenize_vietnamese(doc.page_content)
            self.doc_len[i] = len(tokens)
            all_tokens.extend(tokens)

            # Count terms per document
            unique_tokens = set(tokens)
            for token in unique_tokens:
                if token not in self.doc_frequencies:
                    self.doc_frequencies[token] = 0
                self.doc_frequencies[token] += 1

        # Calculate average document length
        self.avgdl = sum(self.doc_len.values()) / len(self.documents)

        # Calculate IDF cho mỗi term
        N = len(self.documents)
        for term, df in self.doc_frequencies.items():
            self.idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1)

    def _tokenize_vietnamese(self, text: str) -> List[str]:
        """
        Tokenize text tiếng Việt.
        Đơn giản hóa bằng cách split theo space và normalize.
        """
        if not text:
            return []

        # Lowercase và remove special chars
        text = text.lower()
        import re

        text = re.sub(r"[^\w\s]", " ", text)

        # Split và filter empty
        tokens = [t.strip() for t in text.split() if t.strip() and len(t.strip()) > 1]

        return tokens

    def search(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """
        Tìm kiếm documents sử dụng BM25 score.

        Args:
            query: Câu hỏi tìm kiếm
            k: Số kết quả trả về

        Returns:
            List of (document, bm25_score) tuples
        """
        if not self.documents:
            return []

        query_tokens = self._tokenize_vietnamese(query)
        if not query_tokens:
            return []

        scores = []

        for i, doc in enumerate(self.documents):
            score = self._calculate_bm25_score(query_tokens, i)
            if score > 0:  # Chỉ trả về docs có score > 0
                scores.append((doc, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:k]

    def _calculate_bm25_score(self, query_tokens: List[str], doc_idx: int) -> float:
        """Tính BM25 score cho một document."""
        doc = self.documents[doc_idx]
        doc_tokens = self._tokenize_vietnamese(doc.page_content)
        doc_len = self.doc_len[doc_idx]

        if doc_len == 0:
            return 0.0

        # Count term frequencies trong document
        tf = Counter(doc_tokens)
        score = 0.0

        for term in query_tokens:
            if term in tf and term in self.idf:
                # BM25 formula
                term_freq = tf[term]
                idf = self.idf[term]

                numerator = term_freq * (self.k1 + 1)
                denominator = term_freq + self.k1 * (
                    1 - self.b + self.b * (doc_len / self.avgdl)
                )

                score += idf * (numerator / denominator)

        return score


class HybridRetriever:
    """
    Hybrid Retriever kết hợp BM25 và Embedding search.
    Cung cấp cả keyword matching và semantic similarity.
    """

    def __init__(
        self,
        vectorstore: VectorStore,
        default_k: int = 5,
        score_threshold: float = 0.1,  # Lowered from 0.3 to 0.1
        bm25_weight: float = 0.3,
        embedding_weight: float = 0.7,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ):
        """
        Khởi tạo hybrid retriever.

        Args:
            vectorstore: Vector store cho embedding search
            default_k: Số documents mặc định
            score_threshold: Ngưỡng điểm tối thiểu
            bm25_weight: Trọng số cho BM25 score (0-1)
            embedding_weight: Trọng số cho embedding score (0-1)
            bm25_k1: BM25 k1 parameter
            bm25_b: BM25 b parameter
        """
        self.vectorstore = vectorstore
        self.default_k = default_k
        self.score_threshold = score_threshold
        self.bm25_weight = bm25_weight
        self.embedding_weight = embedding_weight

        # Chuẩn hóa weights
        total_weight = bm25_weight + embedding_weight
        if total_weight > 0:
            self.bm25_weight = bm25_weight / total_weight
            self.embedding_weight = embedding_weight / total_weight

        # Khởi tạo BM25 retriever
        self.bm25_retriever = None
        self._initialize_bm25(bm25_k1, bm25_b)

        logger.info(
            f"Khởi tạo HybridRetriever: BM25({self.bm25_weight:.2f}) + Embedding({self.embedding_weight:.2f})"
        )

    def _initialize_bm25(self, k1: float, b: float):
        """Khởi tạo BM25 retriever với documents từ vectorstore."""
        try:
            # Langchain-qdrant không hỗ trợ get all documents
            # BM25 sẽ được lazy initialize khi có documents được add
            logger.info("BM25 sẽ được khởi tạo sau khi có documents")
            self.bm25_retriever = None

        except Exception as e:
            logger.error(f"Lỗi khởi tạo BM25: {e}")
            self.bm25_retriever = None

    def add_documents_to_bm25(self, documents: List[Document]):
        """Thêm documents vào BM25 index."""
        try:
            if not documents:
                return

            if self.bm25_retriever is None:
                # Initialize BM25 with first batch of documents
                self.bm25_retriever = BM25Retriever(documents)
                logger.info(f"Khởi tạo BM25 với {len(documents)} documents")
            else:
                # Extend existing BM25 with new documents
                # Simple approach: recreate BM25 with all documents
                all_docs = self.bm25_retriever.documents + documents
                self.bm25_retriever = BM25Retriever(all_docs)
                logger.info(f"Cập nhật BM25 với {len(all_docs)} documents")

        except Exception as e:
            logger.error(f"Lỗi thêm documents vào BM25: {e}")

    async def hybrid_search(
        self,
        query: str,
        k: Optional[int] = None,
        method: str = "hybrid",  # "hybrid", "bm25_only", "embedding_only"
        filter_dict: Optional[Dict[str, Any]] = None,
        include_scores: bool = True,
    ) -> List[Document]:
        """
        Tìm kiếm hybrid kết hợp BM25 và embedding.

        Args:
            query: Câu hỏi tìm kiếm
            k: Số kết quả trả về
            method: Phương pháp tìm kiếm ("hybrid", "bm25_only", "embedding_only")
            filter_dict: Filters (chỉ áp dụng cho embedding search)
            include_scores: Bao gồm scores trong metadata

        Returns:
            List of documents với scores
        """
        k = k or self.default_k

        try:
            if method == "bm25_only":
                return await self._bm25_search_only(query, k, include_scores)
            elif method == "embedding_only":
                return await self._embedding_search_only(
                    query, k, filter_dict, include_scores
                )
            else:  # hybrid
                return await self._hybrid_search_fusion(
                    query, k, filter_dict, include_scores
                )

        except Exception as e:
            logger.error(f"Lỗi hybrid search: {e}")
            return []

    async def _bm25_search_only(
        self, query: str, k: int, include_scores: bool
    ) -> List[Document]:
        """Tìm kiếm chỉ với BM25."""
        if not self.bm25_retriever:
            logger.warning("BM25 retriever chưa được khởi tạo")
            return []

        results = self.bm25_retriever.search(query, k=k * 2)  # Lấy nhiều hơn để filter

        documents = []
        for doc, score in results:
            if score >= self.score_threshold and len(documents) < k:
                # Create a copy để không modify original
                doc_copy = Document(
                    page_content=doc.page_content, metadata=doc.metadata.copy()
                )
                if include_scores:
                    doc_copy.metadata["bm25_score"] = score
                    doc_copy.metadata["hybrid_score"] = score
                    doc_copy.metadata["retrieval_method"] = "bm25_only"
                documents.append(doc_copy)

        return documents

    async def _embedding_search_only(
        self, query: str, k: int, filter_dict: Optional[Dict], include_scores: bool
    ) -> List[Document]:
        """Tìm kiếm chỉ với embedding."""
        results_with_scores = self.vectorstore.similarity_search_with_score(
            query=query,
            k=k,
            score_threshold=self.score_threshold,
        )

        documents = []
        for doc, score in results_with_scores:
            if include_scores:
                doc.metadata["embedding_score"] = score
                doc.metadata["hybrid_score"] = score
                doc.metadata["retrieval_method"] = "embedding_only"
            documents.append(doc)

        return documents

    async def _hybrid_search_fusion(
        self, query: str, k: int, filter_dict: Optional[Dict], include_scores: bool
    ) -> List[Document]:
        """Tìm kiếm hybrid với score fusion."""
        # 1. BM25 search
        bm25_results = {}
        if self.bm25_retriever:
            bm25_docs = self.bm25_retriever.search(
                query, k=k * 3
            )  # Lấy nhiều để fusion
            for doc, score in bm25_docs:
                # Use content hash as key để match với embedding results
                content_key = hash(doc.page_content[:200])  # Hash of first 200 chars
                bm25_results[content_key] = (doc, score)

        # 2. Embedding search
        embedding_results = {}
        embedding_docs = self.vectorstore.similarity_search_with_score(
            query=query, k=k * 3, score_threshold=0.0
        )

        for doc, score in embedding_docs:
            content_key = hash(doc.page_content[:200])
            embedding_results[content_key] = (doc, score)

        # 3. Score fusion
        fused_scores = []
        all_keys = set(bm25_results.keys()) | set(embedding_results.keys())

        for content_key in all_keys:
            bm25_score = 0.0
            embedding_score = 0.0
            doc = None

            if content_key in bm25_results:
                doc, bm25_score = bm25_results[content_key]

            if content_key in embedding_results:
                doc, embedding_score = embedding_results[content_key]

            if doc is None:
                continue

            # Normalize scores (0-1 range)
            normalized_bm25 = self._normalize_bm25_score(bm25_score)
            normalized_embedding = self._normalize_embedding_score(embedding_score)

            # Weighted fusion
            hybrid_score = (
                self.bm25_weight * normalized_bm25
                + self.embedding_weight * normalized_embedding
            )

            if hybrid_score >= self.score_threshold:
                fused_scores.append((doc, hybrid_score, bm25_score, embedding_score))

        # Sort by hybrid score
        fused_scores.sort(key=lambda x: x[1], reverse=True)

        # Prepare final results
        documents = []
        for doc, hybrid_score, bm25_score, embedding_score in fused_scores[:k]:
            # Create copy để không modify original
            doc_copy = Document(
                page_content=doc.page_content, metadata=doc.metadata.copy()
            )
            if include_scores:
                doc_copy.metadata["hybrid_score"] = hybrid_score
                doc_copy.metadata["bm25_score"] = bm25_score
                doc_copy.metadata["embedding_score"] = embedding_score
                doc_copy.metadata["retrieval_method"] = "hybrid_fusion"
                doc_copy.metadata["bm25_weight"] = self.bm25_weight
                doc_copy.metadata["embedding_weight"] = self.embedding_weight
            documents.append(doc_copy)

        return documents

    def _normalize_bm25_score(self, score: float) -> float:
        """Normalize BM25 score to 0-1 range."""
        # BM25 scores can vary widely, use a reasonable cap
        max_bm25 = 20.0  # Reasonable maximum for BM25
        return min(score / max_bm25, 1.0)

    def _normalize_embedding_score(self, score: float) -> float:
        """Normalize embedding score to 0-1 range."""
        # Embedding scores are typically in [0, 2] range (cosine distance)
        # Convert to similarity: 1 - (score / 2)
        similarity = max(0.0, 1.0 - (score / 2.0))
        return similarity

    async def compare_methods(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        So sánh hiệu suất các phương pháp retrieval.

        Returns:
            Dictionary với kết quả của từng phương pháp
        """
        comparison = {"query": query, "k": k, "methods": {}}

        methods = ["bm25_only", "embedding_only", "hybrid"]

        for method in methods:
            try:
                results = await self.hybrid_search(query, k=k, method=method)

                comparison["methods"][method] = {
                    "count": len(results),
                    "avg_score": (
                        sum(doc.metadata.get("hybrid_score", 0) for doc in results)
                        / len(results)
                        if results
                        else 0
                    ),
                    "documents": [
                        {
                            "content_preview": doc.page_content[:100] + "...",
                            "score": doc.metadata.get("hybrid_score", 0),
                            "bm25_score": doc.metadata.get("bm25_score", 0),
                            "embedding_score": doc.metadata.get("embedding_score", 0),
                            "source": doc.metadata.get("source_file", "unknown"),
                        }
                        for doc in results[:3]  # Top 3 for comparison
                    ],
                }

            except Exception as e:
                comparison["methods"][method] = {"error": str(e)}

        return comparison

    def get_stats(self) -> Dict[str, Any]:
        """Lấy thống kê về hybrid retriever."""
        stats = {
            "retriever_type": "hybrid",
            "weights": {"bm25": self.bm25_weight, "embedding": self.embedding_weight},
            "bm25_initialized": self.bm25_retriever is not None,
            "vectorstore_type": type(self.vectorstore).__name__,
        }

        if self.bm25_retriever:
            stats["bm25_stats"] = {
                "document_count": len(self.bm25_retriever.documents),
                "average_doc_length": self.bm25_retriever.avgdl,
                "vocabulary_size": len(self.bm25_retriever.idf),
            }

        return stats


# Global instance
_hybrid_retriever = None


def get_hybrid_retriever(
    vectorstore: VectorStore,
    default_k: int = 5,
    score_threshold: float = 0.1,  # Lowered from 0.3 to 0.1
    bm25_weight: float = 0.3,
    embedding_weight: float = 0.7,
) -> HybridRetriever:
    """Lấy instance HybridRetriever toàn cục."""
    global _hybrid_retriever
    if _hybrid_retriever is None:
        _hybrid_retriever = HybridRetriever(
            vectorstore=vectorstore,
            default_k=default_k,
            score_threshold=score_threshold,
            bm25_weight=bm25_weight,
            embedding_weight=embedding_weight,
        )
    return _hybrid_retriever
