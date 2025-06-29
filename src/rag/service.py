"""
Dịch vụ RAG nâng cao tích hợp RAGFlow và HuggingFace.
Cung cấp khả năng xử lý tài liệu và truy vấn tiếng Việt tối ưu.
"""

import os
from typing import List, Dict, Any, Optional
from langchain.schema import Document

from .embeddings import get_embedding_service, MultiEmbeddingService, Embeddings
from .chunker import get_vietnamese_chunker
from .hybrid_retriever import get_hybrid_retriever
from core.llm_client import get_llm
from langchain_qdrant import QdrantVectorStore
from core.logging_config import get_logger

logger = get_logger(__name__)


class RAGService:
    """
    Dịch vụ RAG nâng cao với:
    - HuggingFace embeddings tối ưu cho tiếng Việt
    - RAGFlow PDF parsing cải tiến
    - Chunking thông minh với bảo toàn cấu trúc
    - Prompt engineering nâng cao
    """

    def __init__(
        self,
        # Cấu hình embedding
        embedding_provider: str = "huggingface",
        embedding_model: str = "intfloat/multilingual-e5-base",
        embedding_api_key: Optional[str] = None,
        embedding_device: str = "auto",
        # Cấu hình Qdrant vector store
        collection_name: str = "vietnamese_knowledge_base",
        qdrant_url: str = "http://localhost:6333",
        qdrant_api_key: Optional[str] = None,
        # Cấu hình chunking
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        use_ragflow_pdf: bool = True,
        preserve_structure: bool = True,
        # Cấu hình retrieval
        retrieval_k: int = 5,
        score_threshold: float = 0.1,
        use_reranking: bool = False,
    ):
        """
        Khởi tạo dịch vụ RAG nâng cao.

        Args:
            embedding_provider: Loại provider (huggingface hoặc openai)
            embedding_model: Tên model embedding
            embedding_api_key: API key nếu cần
            embedding_device: Thiết bị chạy (cpu/cuda)
            collection_name: Tên collection vector store
            vectorstore_dir: Thư mục lưu vector store
            chunk_size: Kích thước chunk tối đa
            chunk_overlap: Độ chồng lấp chunks
            use_ragflow_pdf: Sử dụng RAGFlow PDF parser
            preserve_structure: Bảo toàn cấu trúc tài liệu
            retrieval_k: Số documents truy vấn
            score_threshold: Ngưỡng độ tương tự tối thiểu
            use_reranking: Sử dụng reranking
        """

        # Lưu cấu hình
        self.config = {
            "embedding": {
                "provider": embedding_provider,
                "model": embedding_model,
                "device": embedding_device,
            },
            "chunking": {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "use_ragflow_pdf": use_ragflow_pdf,
                "preserve_structure": preserve_structure,
            },
            "retrieval": {
                "k": retrieval_k,
                "score_threshold": score_threshold,
                "use_reranking": use_reranking,
            },
        }

        # Khởi tạo các components
        logger.info("Đang khởi tạo các components RAG nâng cao...")

        # 1. Dịch vụ embedding
        self.embedding_service = get_embedding_service(
            provider=embedding_provider,
            model_name=embedding_model,
            api_key=embedding_api_key,
            device=embedding_device,
        )

        # 2. Bộ chia nhỏ tài liệu tiếng Việt
        self.chunker = get_vietnamese_chunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            use_ragflow_pdf=use_ragflow_pdf,
        )

        # 3. Qdrant Vector Store (langchain-qdrant official)
        # Lấy LangChain embeddings object
        if isinstance(self.embedding_service.embeddings, Embeddings):
            langchain_embeddings = self.embedding_service.embeddings.embeddings
        else:
            langchain_embeddings = self.embedding_service.embeddings

        self.vectorstore = QdrantVectorStore.from_existing_collection(
            embedding=langchain_embeddings,
            collection_name=collection_name,
            url=qdrant_url,
            api_key=qdrant_api_key,
        )

        logger.info(f"Using Qdrant vectorstore: {qdrant_url}")
        logger.info(f"Collection: {collection_name}")

        # 4. Hybrid Retriever (BM25 + Embedding) - MAIN RETRIEVER
        self.hybrid_retriever = get_hybrid_retriever(
            vectorstore=self.vectorstore,
            default_k=retrieval_k,
            score_threshold=score_threshold,
        )

        # Note: Regular retriever đã được thay thế bởi hybrid_retriever
        # Hybrid có thể làm tất cả: hybrid, bm25_only, embedding_only

        # 5. LLM client
        self.llm = get_llm()

        # Lưu tùy chọn xử lý
        self.preserve_structure = preserve_structure
        self.use_reranking = use_reranking

        logger.info("Đã khởi tạo dịch vụ RAG nâng cao thành công")
        logger.info(f"Embedding: {embedding_provider} ({embedding_model})")
        logger.info(
            f"Chunking: ragflow_pdf={use_ragflow_pdf}, structure={preserve_structure}"
        )
        logger.info("Hybrid Retrieval: BM25 + Embedding được kích hoạt")

    async def add_documents_from_files(
        self, file_paths: List[str], preserve_structure: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Thêm documents từ files với xử lý nâng cao.

        Args:
            file_paths: Danh sách đường dẫn files
            preserve_structure: Ghi đè cấu hình bảo toàn cấu trúc

        Returns:
            Kết quả xử lý chi tiết
        """
        try:
            # Sử dụng config mặc định nếu không chỉ định
            if preserve_structure is None:
                preserve_structure = self.preserve_structure

            # Xử lý tài liệu nâng cao
            logger.info(
                f"Đang xử lý {len(file_paths)} files với bộ chunker tiếng Việt..."
            )

            chunks = self.chunker.process_multiple_files(
                file_paths=file_paths, preserve_structure=preserve_structure
            )

            if not chunks:
                return {
                    "success": False,
                    "message": "Không có documents nào được xử lý thành công",
                    "files_processed": 0,
                    "chunks_added": 0,
                    "processing_details": [],
                }

            # Thêm vào vector store
            logger.info(f"Đang thêm {len(chunks)} chunks vào vector store...")
            document_ids = self.vectorstore.add_documents(chunks)

            # Initialize BM25 với documents mới
            if self.hybrid_retriever.bm25_retriever is None:
                logger.info("Khởi tạo BM25 retriever với documents mới...")
                self.hybrid_retriever.add_documents_to_bm25(chunks)
            else:
                # Thêm documents mới vào BM25 existing
                self.hybrid_retriever.add_documents_to_bm25(chunks)

            # Tạo chi tiết xử lý
            processing_details = self._generate_processing_details(file_paths, chunks)

            logger.info(
                f"Đã thêm thành công {len(chunks)} chunks từ {len(file_paths)} files"
            )

            return {
                "success": True,
                "message": f"Đã xử lý thành công {len(file_paths)} files → {len(chunks)} chunks",
                "files_processed": len(file_paths),
                "chunks_added": len(chunks),
                "document_ids": document_ids,
                "processing_details": processing_details,
                "advanced_features": {
                    "ragflow_pdf_parsing": self.chunker.use_ragflow_pdf,
                    "structure_preservation": preserve_structure,
                    "vietnamese_optimization": True,
                    "embedding_provider": self.config["embedding"]["provider"],
                },
            }

        except Exception as e:
            logger.error(f"Lỗi trong xử lý documents nâng cao: {e}")
            return {
                "success": False,
                "message": f"Lỗi xử lý documents: {str(e)}",
                "files_processed": 0,
                "chunks_added": 0,
                "processing_details": [],
            }

    async def add_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        preserve_structure: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Thêm raw text với xử lý nâng cao."""
        try:
            if preserve_structure is None:
                preserve_structure = self.preserve_structure

            # Xử lý text nâng cao
            chunks = self.chunker.process_text(
                text=text, metadata=metadata, preserve_structure=preserve_structure
            )

            # Thêm vào vector store
            document_ids = self.vectorstore.add_documents(chunks)

            # Initialize BM25 với documents mới
            if self.hybrid_retriever.bm25_retriever is None:
                logger.info("Khởi tạo BM25 retriever với documents mới...")
                self.hybrid_retriever.add_documents_to_bm25(chunks)
            else:
                # Thêm documents mới vào BM25 existing
                self.hybrid_retriever.add_documents_to_bm25(chunks)

            logger.info(f"Đã thêm text với {len(chunks)} chunks (tối ưu tiếng Việt)")

            return {
                "success": True,
                "message": f"Đã thêm text với {len(chunks)} chunks (nâng cao)",
                "chunks_added": len(chunks),
                "document_ids": document_ids,
                "advanced_features": {
                    "structure_preservation": preserve_structure,
                    "vietnamese_optimization": True,
                    "embedding_provider": self.config["embedding"]["provider"],
                },
            }

        except Exception as e:
            logger.error(f"Lỗi thêm text: {e}")
            return {
                "success": False,
                "message": f"Lỗi thêm text: {str(e)}",
                "chunks_added": 0,
            }

    async def query(
        self,
        question: str,
        k: Optional[int] = None,
        use_reranking: Optional[bool] = None,
        include_sources: bool = True,
        vietnamese_prompt: bool = True,
    ) -> Dict[str, Any]:
        """
        Truy vấn nâng cao với cải thiện cho tiếng Việt.

        Args:
            question: Câu hỏi của người dùng
            k: Số documents truy vấn
            use_reranking: Sử dụng reranking (ghi đè config)
            include_sources: Bao gồm thông tin nguồn
            vietnamese_prompt: Sử dụng prompt tối ưu tiếng Việt

        Returns:
            Kết quả truy vấn nâng cao
        """
        try:
            # Sử dụng config mặc định nếu không chỉ định
            if k is None:
                k = self.config["retrieval"]["k"]
            if use_reranking is None:
                use_reranking = self.use_reranking

            logger.info(f"Đang xử lý truy vấn nâng cao: {question[:100]}...")

            # Retrieval nâng cao
            # Sử dụng hybrid retriever với embedding_only mode để maintain compatibility
            if use_reranking:
                # Đảm bảo k không None
                k_value = k if k is not None else self.config["retrieval"]["k"]
                # Dùng hybrid với embedding_only + manual reranking logic
                candidates = await self.hybrid_retriever.hybrid_search(
                    query=question,
                    k=k_value * 2,  # Lấy nhiều để rerank
                    method="embedding_only",
                    include_scores=True,
                )
                # Simple reranking based on scores
                candidates.sort(
                    key=lambda doc: doc.metadata.get("similarity_score", 0),
                    reverse=True,
                )
                relevant_docs = candidates[:k_value]
            else:
                relevant_docs = await self.hybrid_retriever.hybrid_search(
                    query=question, k=k, method="embedding_only", include_scores=True
                )

            if not relevant_docs:
                return {
                    "success": True,
                    "answer": "Tôi không tìm thấy thông tin liên quan để trả lời câu hỏi này.",
                    "sources": [],
                    "num_sources": 0,
                    "retrieval_method": "advanced",
                    "confidence_score": 0.0,
                }

            # Xây dựng context nâng cao
            context = self._build_vietnamese_context(relevant_docs)

            # Tạo prompt nâng cao
            if vietnamese_prompt:
                prompt = self._build_vietnamese_prompt(question, context, relevant_docs)
            else:
                prompt = self._build_basic_prompt(question, context)

            # Tạo câu trả lời
            answer = await self.llm.generate(prompt)

            # Tính điểm tin cậy
            confidence_score = self._calculate_confidence_score(relevant_docs)

            # Chuẩn bị thông tin nguồn nâng cao
            sources = []
            if include_sources:
                sources = self._extract_vietnamese_sources(relevant_docs)

            logger.info(
                f"Truy vấn nâng cao hoàn thành với {len(relevant_docs)} sources"
            )

            return {
                "success": True,
                "answer": answer.strip(),
                "sources": sources,
                "num_sources": len(relevant_docs),
                "retrieval_method": (
                    "advanced_reranking" if use_reranking else "advanced"
                ),
                "confidence_score": confidence_score,
                "processing_info": {
                    "embedding_provider": self.config["embedding"]["provider"],
                    "reranking_used": use_reranking,
                    "vietnamese_prompt": vietnamese_prompt,
                    "vietnamese_optimized": True,
                },
            }

        except Exception as e:
            logger.error(f"Lỗi trong truy vấn nâng cao: {e}")
            return {
                "success": False,
                "answer": f"Lỗi xử lý câu hỏi: {str(e)}",
                "sources": [],
                "num_sources": 0,
                "confidence_score": 0.0,
            }

    async def hybrid_query(
        self,
        question: str,
        k: Optional[int] = None,
        method: str = "hybrid",  # "hybrid", "bm25_only", "embedding_only"
        include_sources: bool = True,
        vietnamese_prompt: bool = True,
    ) -> Dict[str, Any]:
        """
        Truy vấn hybrid sử dụng cả BM25 và embedding search.

        Args:
            question: Câu hỏi cần trả lời
            k: Số documents truy vấn
            method: Phương pháp ("hybrid", "bm25_only", "embedding_only")
            include_sources: Bao gồm thông tin nguồn
            vietnamese_prompt: Sử dụng prompt tiếng Việt

        Returns:
            Kết quả truy vấn với thông tin về phương pháp đã sử dụng
        """
        try:
            k = k or self.config["retrieval"]["k"]

            # Hybrid retrieval
            relevant_docs = await self.hybrid_retriever.hybrid_search(
                query=question, k=k, method=method, include_scores=True
            )

            if not relevant_docs:
                return {
                    "success": True,
                    "answer": "Xin lỗi, tôi không tìm thấy thông tin liên quan trong cơ sở tri thức để trả lời câu hỏi này.",
                    "sources": [],
                    "num_sources": 0,
                    "retrieval_method": f"hybrid_{method}",
                    "confidence_score": 0.0,
                    "hybrid_info": {
                        "method_used": method,
                        "bm25_available": self.hybrid_retriever.bm25_retriever
                        is not None,
                        "documents_searched": 0,
                    },
                }

            # Xây dựng context
            context = self._build_vietnamese_context(relevant_docs)

            # Tạo prompt
            if vietnamese_prompt:
                prompt = self._build_vietnamese_hybrid_prompt(
                    question, context, relevant_docs, method
                )
            else:
                prompt = self._build_basic_prompt(question, context)

            # Gọi LLM
            answer = await self.llm.generate(prompt)

            # Tính confidence score
            confidence_score = self._calculate_confidence_score(relevant_docs)

            # Chuẩn bị sources
            sources = []
            if include_sources:
                sources = self._extract_vietnamese_sources(relevant_docs)

            # Thống kê hybrid
            hybrid_stats = self._extract_hybrid_stats(relevant_docs, method)

            return {
                "success": True,
                "answer": answer.strip(),
                "sources": sources,
                "num_sources": len(relevant_docs),
                "retrieval_method": f"hybrid_{method}",
                "confidence_score": confidence_score,
                "hybrid_info": {
                    "method_used": method,
                    "bm25_available": self.hybrid_retriever.bm25_retriever is not None,
                    "bm25_weight": hybrid_stats.get("bm25_weight", 0),
                    "embedding_weight": hybrid_stats.get("embedding_weight", 0),
                    "avg_bm25_score": hybrid_stats.get("avg_bm25_score", 0),
                    "avg_embedding_score": hybrid_stats.get("avg_embedding_score", 0),
                    "documents_searched": len(relevant_docs),
                },
                "processing_info": {
                    "embedding_provider": self.config["embedding"]["provider"],
                    "vietnamese_prompt": vietnamese_prompt,
                    "vietnamese_optimized": True,
                    "hybrid_fusion": method == "hybrid",
                },
            }

        except Exception as e:
            logger.error(f"Lỗi trong hybrid query: {e}")
            return {
                "success": False,
                "answer": f"Lỗi xử lý câu hỏi hybrid: {str(e)}",
                "sources": [],
                "num_sources": 0,
                "confidence_score": 0.0,
                "hybrid_info": {"method_used": method, "error": str(e)},
            }

    async def compare_retrieval_methods(
        self, question: str, k: int = 5
    ) -> Dict[str, Any]:
        """
        So sánh hiệu suất của các phương pháp retrieval khác nhau.

        Args:
            question: Câu hỏi để test
            k: Số documents để so sánh

        Returns:
            So sánh chi tiết giữa các phương pháp
        """
        try:
            # So sánh hybrid methods
            hybrid_comparison = await self.hybrid_retriever.compare_methods(question, k)

            # Thêm regular embedding-only retrieval để so sánh
            regular_docs = await self.hybrid_retriever.hybrid_search(
                query=question, k=k, method="embedding_only", include_scores=True
            )

            regular_results = {
                "count": len(regular_docs),
                "avg_score": (
                    sum(doc.metadata.get("similarity_score", 0) for doc in regular_docs)
                    / len(regular_docs)
                    if regular_docs
                    else 0
                ),
                "documents": [
                    {
                        "content_preview": doc.page_content[:100] + "...",
                        "score": doc.metadata.get("similarity_score", 0),
                        "source": doc.metadata.get("source_file", "unknown"),
                    }
                    for doc in regular_docs[:3]
                ],
            }

            hybrid_comparison["methods"]["regular_embedding"] = regular_results

            # Thêm metadata
            hybrid_comparison["comparison_info"] = {
                "question": question,
                "k": k,
                "timestamp": "N/A",  # Could add timestamp
                "recommendation": self._recommend_best_method(
                    hybrid_comparison["methods"]
                ),
            }

            return hybrid_comparison

        except Exception as e:
            logger.error(f"Lỗi so sánh retrieval methods: {e}")
            return {"error": str(e), "question": question, "k": k}

    def _build_vietnamese_hybrid_prompt(
        self, question: str, context: str, documents: List[Document], method: str
    ) -> str:
        """Xây dựng prompt hybrid với thông tin về phương pháp retrieval."""
        doc_info = self._analyze_vietnamese_documents(documents)
        hybrid_info = self._extract_hybrid_stats(documents, method)

        method_description = {
            "hybrid": "Kết hợp BM25 (từ khóa) + Embedding (ngữ nghĩa)",
            "bm25_only": "BM25 - Tìm kiếm theo từ khóa chính xác",
            "embedding_only": "Embedding - Tìm kiếm theo ngữ nghĩa",
        }

        prompt = f"""Bạn là một AI assistant thông minh sử dụng hệ thống tìm kiếm hybrid tiên tiến. Hãy trả lời câu hỏi dựa trên thông tin được tìm kiếm bằng phương pháp {method_description.get(method, method)}.

🔍 **THÔNG TIN TÌM KIẾM:**
- Phương pháp: {method_description.get(method, method)}
- Số nguồn tìm thấy: {len(documents)}
- Độ tin cậy trung bình: {doc_info['avg_confidence']:.3f}
{f"- Trọng số BM25: {hybrid_info.get('bm25_weight', 0):.2f}" if method == "hybrid" else ""}
{f"- Trọng số Embedding: {hybrid_info.get('embedding_weight', 0):.2f}" if method == "hybrid" else ""}

📖 **THÔNG TIN THAM KHẢO:**
{context}

❓ **CÂU HỎI:** {question}

💡 **HƯỚNG DẪN TRẢ LỜI:**
1. Sử dụng thông tin từ kết quả tìm kiếm {method_description.get(method, method)}
2. Trích dẫn nguồn cụ thể khi cần thiết
3. Đánh giá độ tin cậy của thông tin dựa trên phương pháp tìm kiếm
4. Trả lời bằng tiếng Việt tự nhiên và chính xác
5. Nêu rõ nếu thông tin không đủ hoặc cần tìm hiểu thêm

**TRẢ LỜI:**"""

        return prompt

    def _extract_hybrid_stats(
        self, documents: List[Document], method: str
    ) -> Dict[str, Any]:
        """Trích xuất thống kê từ kết quả hybrid retrieval."""
        if not documents:
            return {}

        stats = {}
        bm25_scores = []
        embedding_scores = []

        for doc in documents:
            bm25_score = doc.metadata.get("bm25_score", 0)
            embedding_score = doc.metadata.get("embedding_score", 0)

            if bm25_score > 0:
                bm25_scores.append(bm25_score)
            if embedding_score > 0:
                embedding_scores.append(embedding_score)

            # Lấy weights từ document đầu tiên (nếu có)
            if "bm25_weight" in doc.metadata:
                stats["bm25_weight"] = doc.metadata["bm25_weight"]
            if "embedding_weight" in doc.metadata:
                stats["embedding_weight"] = doc.metadata["embedding_weight"]

        if bm25_scores:
            stats["avg_bm25_score"] = sum(bm25_scores) / len(bm25_scores)
        if embedding_scores:
            stats["avg_embedding_score"] = sum(embedding_scores) / len(embedding_scores)

        return stats

    def _recommend_best_method(self, methods_results: Dict[str, Any]) -> str:
        """Đề xuất phương pháp tốt nhất dựa trên kết quả."""
        if not methods_results:
            return "Không có đủ dữ liệu để đề xuất"

        scores = {}
        for method, result in methods_results.items():
            if isinstance(result, dict) and "avg_score" in result:
                scores[method] = result["avg_score"]

        if not scores:
            return "Không thể đánh giá"

        best_method = max(scores.keys(), key=lambda x: scores[x])
        best_score = scores[best_method]

        recommendations = {
            "hybrid_fusion": "Hybrid fusion - Cân bằng tốt nhất giữa từ khóa và ngữ nghĩa",
            "bm25_only": "BM25 - Tốt cho tìm kiếm từ khóa chính xác và thuật ngữ kỹ thuật",
            "embedding_only": "Embedding - Tốt cho tìm kiếm ngữ nghĩa và từ đồng nghĩa",
            "regular_embedding": "Embedding thông thường - Đơn giản và ổn định",
        }

        return (
            f"{recommendations.get(best_method, best_method)} (Điểm: {best_score:.3f})"
        )

    def _generate_processing_details(
        self, file_paths: List[str], chunks: List[Document]
    ) -> List[Dict]:
        """Tạo thông tin chi tiết về quá trình xử lý."""
        details = []

        # Nhóm chunks theo file nguồn
        file_chunks = {}
        for chunk in chunks:
            source_file = chunk.metadata.get("source_file", "unknown")
            if source_file not in file_chunks:
                file_chunks[source_file] = []
            file_chunks[source_file].append(chunk)

        for file_path in file_paths:
            filename = os.path.basename(file_path)
            file_chunks_list = file_chunks.get(filename, [])

            details.append(
                {
                    "file": filename,
                    "file_path": file_path,
                    "chunks_created": len(file_chunks_list),
                    "extraction_method": (
                        file_chunks_list[0].metadata.get(
                            "extraction_method", "standard"
                        )
                        if file_chunks_list
                        else "failed"
                    ),
                    "chunk_method": (
                        file_chunks_list[0].metadata.get("chunk_method", "standard")
                        if file_chunks_list
                        else "failed"
                    ),
                    "vietnamese_optimized": (
                        file_chunks_list[0].metadata.get("vietnamese_optimized", False)
                        if file_chunks_list
                        else False
                    ),
                }
            )

        return details

    def _build_vietnamese_context(self, documents: List[Document]) -> str:
        """Xây dựng context nâng cao với định dạng tiếng Việt."""
        if not documents:
            return ""

        context_parts = []

        for i, doc in enumerate(documents, 1):
            content = doc.page_content.strip()
            source = doc.metadata.get("source_file", "Không rõ")
            page = doc.metadata.get("page", "N/A")
            score = doc.metadata.get("similarity_score", 0.0)
            extraction_method = doc.metadata.get("extraction_method", "standard")

            # Định dạng context tiếng Việt
            context_parts.append(
                f"📄 **Nguồn {i}:** {source} (Trang {page}) | Độ liên quan: {score:.3f} | Phương pháp: {extraction_method}\n"
                f"{content}\n"
                f"{'─' * 50}"
            )

        return "\n\n".join(context_parts)

    def _build_vietnamese_prompt(
        self, question: str, context: str, documents: List[Document]
    ) -> str:
        """Xây dựng prompt nâng cao tối ưu cho tiếng Việt."""
        # Phân tích documents
        doc_info = self._analyze_vietnamese_documents(documents)

        prompt = f"""Bạn là một AI assistant chuyên nghiệp và thông minh, được thiết kế đặc biệt để hỗ trợ người dùng tiếng Việt. Hãy trả lời câu hỏi dựa trên thông tin được cung cấp.

📋 **THÔNG TIN TÀI LIỆU:**
- Số lượng nguồn: {len(documents)}
- Loại tài liệu: {doc_info['document_types']}
- Phương pháp trích xuất: {doc_info['extraction_methods']}
- Độ tin cậy trung bình: {doc_info['avg_confidence']:.3f}
- Tối ưu hóa tiếng Việt: {doc_info['vietnamese_optimized']}

📖 **THÔNG TIN THAM KHẢO:**
{context}

❓ **CÂU HỎI:** {question}

💡 **HƯỚNG DẪN TRẢ LỜI:**
1. Dựa vào thông tin trên để trả lời chính xác và chi tiết
2. Trích dẫn cụ thể nguồn tài liệu khi cần thiết (ví dụ: "Theo nguồn 1...")
3. Nếu thông tin không đủ, hãy nói rõ giới hạn
4. Trả lời bằng tiếng Việt tự nhiên, dễ hiểu và phù hợp văn hóa
5. Cấu trúc câu trả lời logic, rõ ràng với các ý chính
6. Sử dụng bullet points hoặc đánh số khi cần thiết

**TRẢ LỜI:**"""

        return prompt

    def _build_basic_prompt(self, question: str, context: str) -> str:
        """Xây dựng prompt cơ bản để tương thích."""
        return f"""Bạn là một AI assistant hữu ích. Dựa vào thông tin được cung cấp dưới đây, hãy trả lời câu hỏi một cách chính xác và chi tiết.

THÔNG TIN THAM KHẢO:
{context}

CÂU HỎI: {question}

Hãy trả lời dựa trên thông tin trên, sử dụng tiếng Việt tự nhiên:"""

    def _analyze_vietnamese_documents(
        self, documents: List[Document]
    ) -> Dict[str, Any]:
        """Phân tích documents để tạo context prompt."""
        if not documents:
            return {
                "document_types": "Không có",
                "extraction_methods": "Không có",
                "avg_confidence": 0.0,
                "vietnamese_optimized": False,
            }

        # Trích xuất thông tin documents
        types = set()
        methods = set()
        scores = []
        vietnamese_optimized = False

        for doc in documents:
            # Loại tài liệu từ extension
            source = doc.metadata.get("source_file", "")
            if "." in source:
                ext = source.split(".")[-1].upper()
                types.add(ext)

            # Phương pháp trích xuất
            method = doc.metadata.get("extraction_method", "standard")
            methods.add(method)

            # Điểm tin cậy từ hybrid retriever keys
            score = (
                doc.metadata.get("hybrid_score", 0.0)
                or doc.metadata.get("bm25_score", 0.0)
                or doc.metadata.get("embedding_score", 0.0)
                or doc.metadata.get("similarity_score", 0.0)  # fallback
            )
            scores.append(score)

            # Tối ưu tiếng Việt
            if doc.metadata.get("vietnamese_optimized", False):
                vietnamese_optimized = True

        return {
            "document_types": ", ".join(sorted(types)) or "Hỗn hợp",
            "extraction_methods": ", ".join(sorted(methods)),
            "avg_confidence": sum(scores) / len(scores) if scores else 0.0,
            "vietnamese_optimized": vietnamese_optimized,
        }

    def _calculate_confidence_score(self, documents: List[Document]) -> float:
        """Tính điểm tin cậy tổng thể."""
        if not documents:
            return 0.0

        scores = []
        for doc in documents:
            # Lấy score từ hybrid retriever keys
            score = (
                doc.metadata.get("hybrid_score", 0.0)
                or doc.metadata.get("bm25_score", 0.0)
                or doc.metadata.get("embedding_score", 0.0)
                or doc.metadata.get("similarity_score", 0.0)  # fallback
            )
            scores.append(score)

        return sum(scores) / len(scores) if scores else 0.0

    def _extract_vietnamese_sources(
        self, documents: List[Document]
    ) -> List[Dict[str, Any]]:
        """Trích xuất thông tin nguồn nâng cao cho tiếng Việt."""
        sources = []

        for i, doc in enumerate(documents, 1):
            # Lấy similarity score từ hybrid retriever keys
            similarity_score = (
                doc.metadata.get("hybrid_score", 0.0)
                or doc.metadata.get("bm25_score", 0.0)
                or doc.metadata.get("embedding_score", 0.0)
                or doc.metadata.get("similarity_score", 0.0)  # fallback
            )

            source_info = {
                "index": i,
                "source_file": doc.metadata.get("source_file", "Không rõ"),
                "page": doc.metadata.get("page", "N/A"),
                "similarity_score": similarity_score,
                "chunk_id": doc.metadata.get("chunk_id", i - 1),
                "extraction_method": doc.metadata.get("extraction_method", "standard"),
                "chunk_method": doc.metadata.get("chunk_method", "standard"),
                "vietnamese_optimized": doc.metadata.get("vietnamese_optimized", False),
                "retrieval_method": doc.metadata.get("retrieval_method", "unknown"),
                "hybrid_score": doc.metadata.get("hybrid_score", 0.0),
                "bm25_score": doc.metadata.get("bm25_score", 0.0),
                "embedding_score": doc.metadata.get("embedding_score", 0.0),
                "content_preview": (
                    doc.page_content[:200] + "..."
                    if len(doc.page_content) > 200
                    else doc.page_content
                ),
            }
            sources.append(source_info)

        return sources

    def get_system_info(self) -> Dict[str, Any]:
        """Lấy thông tin toàn diện về hệ thống RAG nâng cao."""
        vectorstore_info = {
            "type": "langchain_qdrant",
            "collection_name": self.vectorstore.collection_name,
            "url": "http://localhost:6333",  # Default Qdrant URL
        }

        return {
            "service_type": "qdrant_rag_vietnamese",
            "configuration": self.config,
            "components": {
                "embedding_service": self.embedding_service.get_info(),
                "chunker": {
                    "type": "vietnamese_optimized",
                    "ragflow_pdf": self.chunker.use_ragflow_pdf,
                    "chunk_size": self.chunker.chunk_size,
                    "chunk_overlap": self.chunker.chunk_overlap,
                },
                "vectorstore": vectorstore_info,
                "llm_info": self.llm.get_provider_info(),
            },
            "features": {
                "vietnamese_optimization": True,
                "ragflow_pdf_parsing": self.chunker.use_ragflow_pdf,
                "structure_preservation": self.preserve_structure,
                "reranking_support": True,
                "advanced_prompting": True,
                "huggingface_embedding": self.config["embedding"]["provider"]
                == "huggingface",
                "qdrant_vectorstore": True,
            },
        }

    async def clear_knowledge_base(self) -> Dict[str, Any]:
        """Xóa knowledge base."""
        try:
            # Langchain-qdrant doesn't have clear_collection method
            # We'll need to use the underlying client if needed
            logger.warning(
                "Clear collection not implemented for langchain-qdrant. Please manually clear Qdrant collection."
            )

            return {
                "success": True,
                "message": "Cần xóa collection Qdrant thủ công. Collection: "
                + self.vectorstore.collection_name,
                "service_type": "qdrant_rag_vietnamese",
                "collection_name": self.vectorstore.collection_name,
            }
        except Exception as e:
            logger.error(f"Lỗi xóa knowledge base: {e}")
            return {"success": False, "message": f"Lỗi: {str(e)}"}

    async def search_only(
        self,
        question: str,
        k: Optional[int] = None,
        method: str = "hybrid",  # "hybrid", "bm25_only", "embedding_only"
        include_sources: bool = True,
        score_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Chỉ tìm kiếm documents liên quan - KHÔNG gọi LLM (nhanh!).

        Args:
            question: Câu hỏi tìm kiếm
            k: Số documents trả về
            method: Phương pháp tìm kiếm
            include_sources: Bao gồm thông tin nguồn chi tiết
            score_threshold: Ngưỡng điểm tối thiểu

        Returns:
            Chỉ kết quả tìm kiếm, không có LLM answer
        """
        try:
            k = k or self.config["retrieval"]["k"]

            logger.info(f"Đang tìm kiếm (search-only): {question[:100]}...")

            # Chỉ retrieval - KHÔNG gọi LLM
            relevant_docs = await self.hybrid_retriever.hybrid_search(
                query=question, k=k, method=method, include_scores=True
            )

            # Filter by score threshold if provided
            if score_threshold:
                filtered_docs = []
                for doc in relevant_docs:
                    score = (
                        doc.metadata.get("hybrid_score", 0.0)
                        or doc.metadata.get("bm25_score", 0.0)
                        or doc.metadata.get("embedding_score", 0.0)
                        or doc.metadata.get("similarity_score", 0.0)
                    )
                    if score >= score_threshold:
                        filtered_docs.append(doc)
                relevant_docs = filtered_docs

            # Tính confidence score
            confidence_score = self._calculate_confidence_score(relevant_docs)

            # Chuẩn bị sources
            sources = []
            if include_sources:
                sources = self._extract_vietnamese_sources(relevant_docs)

            # Hybrid stats
            hybrid_stats = self._extract_hybrid_stats(relevant_docs, method)

            logger.info(f"Tìm kiếm hoàn thành với {len(relevant_docs)} documents")

            return {
                "success": True,
                "documents": relevant_docs,  # Raw documents
                "sources": sources,  # Formatted sources
                "num_sources": len(relevant_docs),
                "search_method": method,
                "confidence_score": confidence_score,
                "query": question,
                "k_requested": k,
                "score_threshold": score_threshold,
                "processing_info": {
                    "retrieval_only": True,
                    "llm_used": False,
                    "hybrid_method": method,
                    "embedding_provider": self.config["embedding"]["provider"],
                },
                "hybrid_stats": {
                    "method_used": method,
                    "bm25_available": self.hybrid_retriever.bm25_retriever is not None,
                    "bm25_weight": hybrid_stats.get("bm25_weight", 0),
                    "embedding_weight": hybrid_stats.get("embedding_weight", 0),
                    "avg_bm25_score": hybrid_stats.get("avg_bm25_score", 0),
                    "avg_embedding_score": hybrid_stats.get("avg_embedding_score", 0),
                },
            }

        except Exception as e:
            logger.error(f"Lỗi trong search-only: {e}")
            return {
                "success": False,
                "message": f"Lỗi tìm kiếm: {str(e)}",
                "documents": [],
                "sources": [],
                "num_sources": 0,
                "confidence_score": 0.0,
            }


# Global instance
_advanced_rag_service = None


def get_rag_service(**kwargs) -> RAGService:
    """Lấy instance service RAG."""
    global _advanced_rag_service
    if _advanced_rag_service is None:
        _advanced_rag_service = RAGService(**kwargs)
    return _advanced_rag_service
