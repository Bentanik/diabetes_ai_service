from typing import List, Dict, Any, Optional
import logging
from core.embedding import RerankModel, EmbeddingModel
from core.llm import QwenLLM
from rag.retrieval.base import BaseRetriever
from rag.vector_store import VectorStoreManager
import asyncio

logger = logging.getLogger(__name__)

class Retriever(BaseRetriever):
    def __init__(
        self,
        collections: List[str],
        vector_store_manager: VectorStoreManager = None,
        rerank_model: Optional[RerankModel] = None,
        embedding_model: Optional[EmbeddingModel] = None,
        llm: Optional[QwenLLM] = None,
        external_synonyms: Optional[Dict[str, List[str]]] = None,
        use_rerank: bool = True,
        initial_top_k: int = 100,
        top_k: int = 10
    ):
        if not collections:
            raise ValueError("collections không được rỗng.")

        self.collections = collections
        self.vector_store = vector_store_manager or VectorStoreManager()
        self.rerank_model = rerank_model
        self.embedding_model = embedding_model
        self.llm = llm or QwenLLM()
        self.external_synonyms = external_synonyms or {}
        self.use_rerank = use_rerank
        self.initial_top_k = initial_top_k
        self.top_k = top_k

    async def _get_synonyms(self, query_text: str) -> List[str]:
        """Dùng LLM hoặc external_synonyms để sinh từ đồng nghĩa"""
        synonyms = [query_text]
        
        # Ưu tiên external_synonyms
        for key, syn_list in self.external_synonyms.items():
            if key.lower() in query_text.lower():
                synonyms.extend(syn_list)
        
        # Nếu có LLM, sinh thêm synonym
        if self.llm:
            prompt = f"""
            Cho câu hỏi: "{query_text}"
            Hãy liệt kê các từ đồng nghĩa hoặc cách diễn đạt tương tự trong tiếng Việt, đặc biệt trong lĩnh vực y tế. 
            Trả về danh sách dạng: synonym1, synonym2, synonym3
            """
            llm_response = await self.llm.generate(prompt)
            if llm_response:
                synonyms.extend([s.strip() for s in llm_response.split(",") if s.strip()])
        
        return list(set(synonyms))  # Loại bỏ trùng lặp

    async def _decompose_query(self, query_text: str) -> List[str]:
        """Dùng LLM để phân tích query thành sub-queries"""
        if "là gì" not in query_text.lower():
            return [query_text]
        
        prompt = f"""
        Cho câu hỏi: "{query_text}"
        Hãy phân tích thành các câu hỏi phụ liên quan, ví dụ: khái niệm, các loại, triệu chứng, nguyên nhân, cách điều trị.
        Trả về danh sách dạng: sub_query1, sub_query2, sub_query3
        """
        llm_response = await self.llm.generate(prompt)
        if llm_response:
            return [s.strip() for s in llm_response.split(",") if s.strip()]
        # Fallback nếu LLM thất bại
        base_term = query_text.replace("là gì", "").strip()
        return [
            query_text,
            f"khái niệm {base_term}",
            f"các loại {base_term}",
            f"triệu chứng {base_term}",
            f"nguyên nhân {base_term}"
        ]


    async def retrieve(
        self,
        query_vector: List[float],
        query_text: str = None,
        score_threshold: float = 0.05,
        **filters
    ) -> List[Dict[str, Any]]:
        if not query_vector:
            raise ValueError("query_vector không được rỗng.")

        try:
            logger.info(f"Tìm kiếm với {len(self.collections)} collections")
            logger.info(f"initial_top_k: {self.initial_top_k}, top_k: {self.top_k}")

            expanded_query = query_text
            if query_text:
                synonyms = await self._get_synonyms(query_text)
                expanded_query = " ".join(synonyms)
                logger.info(f"Query expanded: {expanded_query}")

            sub_queries = await self._decompose_query(query_text) if query_text else [query_text]
            logger.info(f"Sub-queries: {sub_queries}")

            # Song song hóa embedding và vector search
            tasks = []
            for sub_q in sub_queries:
                sub_vector = await self.embedding_model.embed(sub_q) if self.embedding_model and sub_q else query_vector
                tasks.append(self.vector_store.search_async(
                    collections=self.collections,
                    query_vector=sub_vector,
                    top_k=self.initial_top_k,
                    score_threshold=0.1,  # Tăng để lọc sớm
                    **filters
                ))
            vector_results_list = await asyncio.gather(*tasks)

            all_results = []
            seen_contents = set()
            for vector_results in vector_results_list:
                for hits in vector_results.values():
                    for hit in hits:
                        content = hit["payload"].get("content", "").strip()
                        if content and content not in seen_contents:
                            all_results.append({
                                "id": hit["id"],
                                "content": content,
                                "score": hit["score"],
                                "payload": hit["payload"]
                            })
                            seen_contents.add(content)

            if not all_results:
                logger.warning("Không tìm thấy chunk nào")
                return []

            unique_results = {item["id"]: item for item in all_results}.values()

            if self.use_rerank and self.rerank_model and query_text and len(unique_results) > 1:
                logger.info("Bắt đầu re-rank...")
                documents = [item["content"] for item in unique_results]
                ranked_pairs = await self.rerank_model.rerank(
                    query=expanded_query,
                    documents=documents,
                    top_k=len(documents)
                )
                id_to_item = {item["id"]: item for item in unique_results}
                final = []
                for doc, score in ranked_pairs:
                    matching_items = [item for item in unique_results if item["content"] == doc]
                    for item in matching_items:
                        if item["id"] in id_to_item and score >= score_threshold:
                            new_item = id_to_item[item["id"]].copy()
                            new_item["score"] = score
                            new_item["score_type"] = "reranked"
                            final.append(new_item)
                final.sort(key=lambda x: x["score"], reverse=True)
                logger.info(f"Re-rank xong, trả về {len(final[:self.top_k])} kết quả")
                return final[:self.top_k]

            list(unique_results).sort(key=lambda x: x["score"], reverse=True)
            filtered_results = [r for r in unique_results if r["score"] >= score_threshold]
            logger.info(f"Fallback: Trả về {len(filtered_results[:self.top_k])} kết quả")
            return filtered_results[:self.top_k]

        except Exception as e:
            logger.error(f"Lỗi trong retrieve: {e}", exc_info=True)
            return []

    async def close(self):
        """Đóng client của LLM"""
        if self.llm:
            await self.llm.close()