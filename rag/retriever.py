from typing import List, Dict, Any, Optional
import asyncio
import json
from rag.vector_store import VectorStoreOperations
from rag.re_ranker import Reranker
from utils import get_logger

logger = get_logger(__name__)

class Retriever:
    def __init__(self):
        self.logger = get_logger(__name__)
        self.vector_store = VectorStoreOperations.get_instance()
        self.reranker = Reranker()
        # giả sử bạn có scorer, nếu không thì comment dòng dưới
        # from rag.analysis import DiabetesScorer
        # self.scorer = DiabetesScorer()

    async def initialize(self):
        """
        Khởi tạo async cho reranker.
        """
        await self.reranker._ensure_initialized()

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        rerank_top_n: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        min_score: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Truy vấn và rerank documents.

        Args:
            query (str): Câu truy vấn.
            top_k (int): Số lượng kết quả ban đầu từ vector store.
            rerank_top_n (int): Số lượng kết quả sau rerank.
            filter (Optional[Dict[str, Any]]): Bộ lọc metadata.
            min_score (float): Ngưỡng điểm tối thiểu.

        Returns:
            List[Dict[str, Any]]: Danh sách kết quả với metadata và điểm.
        """
        try:
            await self.initialize()
            results = await self.vector_store.search(
                query_text=query,
                top_k=top_k,
                filter=filter,
                min_score=min_score
            )
            self.logger.info(f"Retrieved {len(results)} initial results for query: {query[:50]}...")

            if not results:
                return []

            documents = [result.text for result in results]
            reranked_docs = await self.reranker.rerank(query=query, documents=documents, top_k=rerank_top_n)

            reranked_results = []
            for reranked_doc in reranked_docs:
                for result in results:
                    if result.text == reranked_doc['text']:
                        result_dict = result.dict()
                        result_dict["score"] = float(reranked_doc['score'])
                        # Lọc theo điểm tối thiểu
                        if result_dict["score"] >= min_score:
                            reranked_results.append(result_dict)
                        break

            self.logger.info(f"Reranked to {len(reranked_results)} results")
            return reranked_results[:rerank_top_n]
        except Exception as e:
            self.logger.error(f"Lỗi khi truy vấn vector store: {str(e)}", exc_info=True)
            raise

async def main():
    retriever = Retriever()
    query = "Bệnh tiểu đường là gì?"
    output_file = "output_file.json"
    try:
        results = await retriever.retrieve(
            query=query,
            top_k=50,
            rerank_top_n=5,
            filter=None,
            min_score=0.5
        )

        # Lưu file json, kết quả đã là list dict nên json.dump sẽ chạy tốt
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"Kết quả tìm kiếm được lưu tại: {output_file}")

        # In ra console test
        for i, res in enumerate(results, 1):
            print(f"{i}. Text: {res.get('text', 'No text')} - Score: {res.get('score', 0):.3f}")

    except Exception as e:
        logger.error(f"Lỗi khi chạy main: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main())
