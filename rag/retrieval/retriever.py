from typing import List, Dict, Callable, Awaitable
from .base import BaseRetriever
from rag.vector_store.manager import VectorStoreManager

class Retriever(BaseRetriever):
    def __init__(
        self,
        collections: List[str],
        embed_fn: Callable[[str], Awaitable[List[float]]],
    ):
        self.manager = VectorStoreManager()
        self.collections = collections
        self.embed_fn = embed_fn

    async def retrieve(self, query: str, top_k: int = 5) -> Dict[str, List[Dict]]:
        query_vector = await self.embed_fn(query)
        results = await self.manager.search_async(self.collections, query_vector, top_k)
        return results
