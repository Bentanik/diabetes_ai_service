from typing import List, Dict
from qdrant_client.http import models
import uuid
from .client import VectorStoreClient
import asyncio

class VectorStoreManager:
    def __init__(self):
        self.client = VectorStoreClient().connection

    async def create_collection_async(self, name: str, size: int = 768, distance: str = "Cosine") -> bool:
        """Tạo collection nếu chưa tồn tại, trả về True nếu tạo mới, False nếu đã tồn tại"""
        def _create():
            existing = [c.name for c in self.client.get_collections().collections]
            if name in existing:
                return False
            self.client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(size=size, distance=distance)
            )
            return True
        return await asyncio.to_thread(_create)

    async def delete_collection_async(self, name: str) -> None:
        """Xóa collection"""
        await asyncio.to_thread(self.client.delete_collection, name)

    async def insert_async(self, name: str, embeddings: List[List[float]], payloads: List[Dict] = None) -> List[str]:
        """Insert embeddings vào collection"""
        def _insert():
            ids = [str(uuid.uuid4()) for _ in embeddings]
            self.client.upsert(
                collection_name=name,
                points=models.Batch(
                    ids=ids,
                    vectors=embeddings,
                    payloads=payloads or [{} for _ in embeddings]
                )
            )
            return ids
        return await asyncio.to_thread(_insert)

    async def search_async(self, collections: List[str], query_vector: List[float], top_k: int = 5, search_accuracy: float = 0.7) -> Dict[str, List[Dict]]:
        """Search vector trong nhiều collection cùng lúc"""
        def _search():
            results = {}
            for col in collections:
                res = self.client.search(
                    collection_name=col,
                    query_vector=query_vector,
                    limit=top_k,
                    score_threshold=search_accuracy
                )
                results[col] = [{"id": r.id, "payload": r.payload, "score": r.score} for r in res]
            return results

        return await asyncio.to_thread(_search)
