import asyncio
from typing import List, Dict
from core.embedding import EmbeddingModel
from core.llm import QwenLLM
from .retriever import Retriever
from app.database.models import ChatHistoryModel
from sklearn.metrics.pairwise import cosine_similarity
from utils.cache import LRUCache
from utils import get_logger

logger = get_logger(__name__)

class MultiQueryRetriever:
    def __init__(self, embedding_model: EmbeddingModel, llm: QwenLLM):
        self.embedding_model = embedding_model
        self.llm = llm
        self.embedding_cache = LRUCache(maxsize=500, ttl=3600)

    async def get_embedding(self, text: str) -> List[float]:
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        emb = await self.embedding_model.embed(text)
        self.embedding_cache[text] = emb
        return emb

    async def should_rewrite(self, query: str, histories: List[ChatHistoryModel]) -> bool:
        pronouns = ["nó", "vậy", "đó", "kia", "trên", "dưới", "trước", "sau", "ý", "cái"]
        return len(histories) >= 2 and any(p in query.lower() for p in pronouns)

    def reciprocal_rank_fusion(self, results: List[List[Dict]], k: int = 60) -> List[Dict]:
        scores = {}
        id_to_doc = {}
        for res_list in results:
            for rank, item in enumerate(res_list):
                doc_id = item["id"]
                id_to_doc[doc_id] = item
                scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [id_to_doc[_id] for _id in sorted_ids]

    async def filter_relevant(self, query: str, contexts: List[Dict], threshold: float = 0.5) -> List[Dict]:
        if not contexts:
            return []
        try:
            contents = [hit["payload"].get("content", "") for hit in contexts]
            embs = await self.embedding_model.embed_batch([query] + contents)
            sims = cosine_similarity([embs[0]], embs[1:])[0]
            return [ctx for ctx, sim in zip(contexts, sims) if sim >= threshold]
        except Exception as e:
            logger.warning(f"Filtering failed: {e}")
            return contexts

    async def retrieve(
        self,
        query: str,
        histories: List[ChatHistoryModel],
        retriever: Retriever,
        top_k_per_query: int = 2,
        final_top_k: int = 8  # ← tăng để có nhiều context hơn
    ) -> List[str]:
        if not query.strip():
            return []

        # 1. Viết lại nếu cần (giả sử bạn có Reflection, nếu không bỏ phần này)
        rewritten_query = query
        if await self.should_rewrite(query, histories):
            # Dùng Reflection ở đây nếu có
            pass

        subqueries = [rewritten_query]

        # 2. Async retrieval
        async def _task(q):
            try:
                q_emb = await self.get_embedding(q)
                return await retriever.retrieve(q_emb, top_k=top_k_per_query)
            except Exception as e:
                logger.warning(f"Retrieval failed for '{q}': {e}")
                return []

        tasks = [_task(q) for q in subqueries]
        results_list = await asyncio.gather(*tasks)
        valid_results = [r for r in results_list if r]

        if not valid_results:
            return []

        # 3. RRF + filter
        fused = self.reciprocal_rank_fusion(valid_results)
        filtered = await self.filter_relevant(query, fused, 0.5)

        # 4. Làm sạch và trả về
        contents = []
        for hit in filtered[:final_top_k]:
            content = hit["payload"].get("content", "").strip()
            if content:
                content = content.replace("[HEADING]", "### ").replace("[/HEADING]", "\n")
                content = content.replace("[SUBHEADING]", "#### ").replace("[/SUBHEADING]", "\n")
                contents.append(content)
        return contents