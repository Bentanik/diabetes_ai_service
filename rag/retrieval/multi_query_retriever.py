from typing import List, Dict, Any
import re
import logging

from core.embedding import EmbeddingModel, RerankModel
from .retriever import Retriever as SingleQueryRetriever


logger = logging.getLogger(__name__)


class MultiQueryRetriever:
    def __init__(
        self,
        collections: List[str],
        llm,
        embedding_model: EmbeddingModel,
        rerank_model: RerankModel,
        num_queries: int = 5,
        initial_top_k: int = 20,
        top_k: int = 8
    ):
        if not collections:
            raise ValueError("collections không được rỗng.")
        if not llm:
            raise ValueError("llm không được rỗng.")
        if not embedding_model:
            raise ValueError("embedding_model không được rỗng.")

        self.collections = collections
        self.llm = llm
        self.embedding_model = embedding_model
        self.rerank_model = rerank_model
        self.num_queries = num_queries
        self.initial_top_k = initial_top_k
        self.top_k = top_k

        self.base_retriever = SingleQueryRetriever(
            collections=collections,
            rerank_model=rerank_model,
            use_rerank=True,
            initial_top_k=initial_top_k,
            top_k=initial_top_k
        )

    async def generate_queries(self, original_query: str) -> List[str]:
        """
        Dùng LLM để sinh query con thông minh, không cứng
        """
        cleaned_query = re.sub(r'[?!.]+', '', original_query).strip()
        if not cleaned_query:
            cleaned_query = original_query

        prompt = f"""
        Hãy phân tích câu hỏi sau và tạo ra {self.num_queries} câu hỏi con giúp tìm kiếm thông tin toàn diện.
        Mỗi câu hỏi nên tập trung vào một khía cạnh quan trọng: định nghĩa, loại, triệu chứng, nguyên nhân, điều trị, so sánh, v.v.
        Chỉ trả về mỗi câu hỏi trên một dòng, không thêm số thứ tự, không giải thích.

        Câu hỏi gốc: "{original_query}"

        Ví dụ với "Tiểu đường có mấy loại?":
        Tiểu đường có bao nhiêu loại?
        Khái niệm ngắn gọn về từng loại tiểu đường
        Sự khác biệt chính giữa các loại tiểu đường
        Loại tiểu đường nào phổ biến nhất?
        Yếu tố nào quyết định loại tiểu đường?

        Trả lời:
        """
        try:
            response = await self.llm.generate(
                prompt=prompt,
                temperature=0.5,
            )

            lines = [line.strip().strip('- ').strip('* ').strip() for line in response.strip().split('\n') if line.strip()]
            queries = [q for q in lines if q and not q.lower().startswith(("trả lời", "ví dụ"))]
            queries = queries[:self.num_queries]

            logger.info(f"LLM sinh {len(queries)} query con: {queries}")
            return queries

        except Exception as e:
            logger.error(f"Lỗi khi sinh query bằng LLM: {e}")
            base = re.sub(r'[?!.]+', '', original_query).strip()
            if not base:
                base = "bệnh lý y khoa"

            return [
                f"{base} có bao nhiêu loại?",
                f"Khái niệm ngắn gọn về từng loại {base}",
                f"Sự khác biệt chính giữa các loại {base}",
                f"Loại nào phổ biến nhất?",
                f"Yếu tố nào quyết định loại?"
            ][:self.num_queries]

    async def retrieve(
        self,
        query_text: str,
        score_threshold: float = 0.0,
        **filters
    ) -> List[Dict[str, Any]]:
        """
        Tìm kiếm với nhiều query, gộp tất cả kết quả, loại trùng, trả về top_k
        Không ưu tiên nhóm, không chọn 1 từ mỗi nhóm
        """
        try:
            logger.info(f"Bắt đầu multi-query retrieval cho: '{query_text}'")
            generated_queries = await self.generate_queries(query_text)
            logger.info(f"Tạo {len(generated_queries)} query con")

            all_results = []
            seen_contents = set()

            for i, q in enumerate(generated_queries):
                logger.info(f"[Query {i+1}/{len(generated_queries)}] {q}")

                q_vector = await self.embedding_model.embed(q.lower())
                results = await self.base_retriever.retrieve(
                    q_vector,
                    query_text=q,
                    score_threshold=0.0,
                    **filters
                )

                logger.info(f"  → Tìm thấy {len(results)} kết quả")

                for r in results:
                    content = r["content"]
                    if content not in seen_contents:
                        seen_contents.add(content)
                        all_results.append(r)

            logger.info(f"Tổng: {len(all_results)} kết quả sau khi gộp (loại trùng)")

            if not all_results:
                return []

            sorted_results = sorted(all_results, key=lambda x: x["score"], reverse=True)
            final_results = [r for r in sorted_results if r["score"] >= score_threshold][:self.top_k]

            logger.info(f"Trả về {len(final_results)} kết quả cuối cùng")
            for r in final_results:
                logger.info(f"Điểm: {r['score']:.3f} | Nội dung: {r['content'][:100]}...")

            return final_results

        except Exception as e:
            logger.error(f"Lỗi trong multi-query retrieval: {e}", exc_info=True)
            return []