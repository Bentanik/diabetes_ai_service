from typing import List, Dict, Any, Optional
from rag.vector_store import VectorStoreOperations
from rag.re_ranker import Reranker, RerankResult
from utils import get_logger


# class Retriever:
#     def __init__(self):
#         self.logger = get_logger(__name__)
#         self.vector_store = VectorStoreOperations.get_instance()

#     async def retrieve(
#         self,
#         query: str,
#         top_k: int = 5,
#         collection_names: Optional[List[str]] = None,
#         filter: Optional[Dict[str, Any]] = None,
#         min_score: float = 0.5,
#     ) -> List[SearchResult]:
#         try:
#             results = await self.vector_store.search(
#                 query_text=query,
#                 top_k=top_k,
#                 collection_names=collection_names,
#                 filter=filter,
#                 min_score=min_score,
#             )

#             return results
#         except Exception as e:
#             self.logger.error(f"Lỗi khi truy vấn vector store: {str(e)}", exc_info=True)
#             raise


class Retriever:
    def __init__(self):
        self.logger = get_logger(__name__)
        self.vector_store = VectorStoreOperations.get_instance()
        self.reranker = Reranker()

    async def initialize(self):
        await self.reranker._ensure_initialized()

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        rerank_top_n: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        min_score: float = 0.5,
    ) -> List[RerankResult]:
        try:
            await self.initialize()
            results = await self.vector_store.search(
                query_text=query, top_k=top_k, filter=filter, min_score=min_score
            )
            self.logger.info(
                f"Retrieved {len(results)} initial results for query: {query[:50]}..."
            )

            if not results:
                return []

            documents = [result.text for result in results]
            reranked_docs = await self.reranker.rerank(
                query=query, documents=documents, top_k=rerank_top_n
            )

            text_to_result = {r.text: r for r in results}
            reranked_results: List[RerankResult] = []

            for reranked_doc in reranked_docs:
                source = text_to_result.get(reranked_doc.text)
                if source and reranked_doc.score >= min_score:
                    reranked_results.append(
                        RerankResult(
                            text=reranked_doc.text,
                            score=reranked_doc.score,
                        )
                    )

            self.logger.info(f"Reranked to {len(reranked_results)} results")
            return reranked_results[:rerank_top_n]
        except Exception as e:
            self.logger.error(f"Lỗi khi truy vấn vector store: {str(e)}", exc_info=True)
            raise


# async def main():
#     import json

#     retriever = Retriever()
#     query = "Bệnh tiểu đường có mấy loại?"
#     output_file = "output_file.json"
#     try:
#         results = await retriever.retrieve(
#             query=query, top_k=50, rerank_top_n=5, filter=None, min_score=0.5
#         )

#         # Lưu file json, kết quả đã là list dict nên json.dump sẽ chạy tốt
#         with open(output_file, "w", encoding="utf-8") as f:
#             json.dump([r.dict() for r in results], f, ensure_ascii=False, indent=2)

#         logger.info(f"Kết quả tìm kiếm được lưu tại: {output_file}")
#     except Exception as e:
#         logger.error(f"Lỗi khi chạy main: {str(e)}", exc_info=True)
#         raise


# if __name__ == "__main__":
#     asyncio.run(main())
