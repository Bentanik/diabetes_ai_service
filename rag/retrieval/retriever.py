from typing import List, Dict, Any
import logging

from rag.retrieval.base import BaseRetriever
from rag.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class Retriever(BaseRetriever):
    def __init__(
        self,
        collections: List[str],
        vector_store_manager: VectorStoreManager = None,
    ):
        if not collections:
            raise ValueError("collections không được rỗng.")

        self.collections = collections
        self.vector_store = vector_store_manager or VectorStoreManager()

    async def retrieve(
        self,
        query_vector: List[float],
        top_k: int = 5,
        score_threshold: float = 0.6,
        **filters
    ) -> List[Dict[str, Any]]:
        """
        Tìm kiếm trong vector store.
        
        :param query_vector: Vector đã được embed từ bên ngoài.
        :param top_k: Số lượng kết quả.
        :param score_threshold: Ngưỡng điểm (0~1).
        :param filters: Các điều kiện lọc, ví dụ: document_is_active=True, metadata__is_active=True
        :return: Danh sách kết quả: [{id, payload, score, collection}]
        """
        if not query_vector:
            raise ValueError("query_vector không được rỗng.")
        if top_k <= 0:
            raise ValueError("top_k phải > 0.")

        try:
            search_results = await self.vector_store.search_async(
                collections=self.collections,
                query_vector=query_vector,
                top_k=top_k,
                score_threshold=score_threshold,
                **filters
            )

            all_results = []
            for col, hits in search_results.items():
                for hit in hits:
                    all_results.append({
                        "collection": col,
                        "id": hit["id"],
                        "payload": hit["payload"],
                        "score": hit["score"]
                    })

            all_results.sort(key=lambda x: x["score"], reverse=True)
            return all_results
        except Exception as e:
            raise