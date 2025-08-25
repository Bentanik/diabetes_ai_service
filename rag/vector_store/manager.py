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

    async def search_async(
        self,
        collections: List[str],
        query_vector: List[float],
        top_k: int = 5,
        search_accuracy: float = 0.7,
        **filters
    ) -> Dict[str, List[Dict]]:
        def _search():
            results = {}
            for col in collections:
                must_conditions = []
                for key, value in filters.items():
                    field_key = key.replace("__", ".")
                    must_conditions.append(
                        models.FieldCondition(
                            key=field_key,
                            match=models.MatchValue(value=value)
                        )
                    )

                query_filter = models.Filter(must=must_conditions) if must_conditions else None

                res = self.client.search(
                    collection_name=col,
                    query_vector=query_vector,
                    limit=top_k,
                    score_threshold=search_accuracy,
                    query_filter=query_filter
                )

                results[col] = [
                    {
                        "id": r.id,
                        "payload": r.payload,
                        "score": r.score
                    }
                    for r in res
                ]
            return results

        return await asyncio.to_thread(_search)

    async def delete_by_metadata_async(self, collection_name: str, **conditions) -> None:
        if not conditions:
            raise ValueError("Phải có ít nhất một điều kiện để xóa.")

        def _delete():
            must_conditions = []
            for key, value in conditions.items():
                field_key = key.replace("__", ".")
                must_conditions.append(
                    models.FieldCondition(
                        key=field_key,
                        match=models.MatchValue(value=value)
                    )
                )

            query_filter = models.Filter(must=must_conditions)

            self.client.delete(
                collection_name=collection_name,
                points_selector=models.FilterSelector(filter=query_filter)
            )

        await asyncio.to_thread(_delete)

    async def update_payload_async(
        self,
        collection_name: str,
        payload_updates: Dict,
        point_id: str = None,
        **filter_conditions
    ) -> None:
        if not point_id and not filter_conditions:
            raise ValueError("Phải cung cấp 'point_id' hoặc ít nhất một điều kiện filter.")

        def _update():
            if point_id:
                self.client.set_payload(
                    collection_name=collection_name,
                    payload=payload_updates,
                    points=[point_id]
                )
            else:
                must_conditions = []
                for key, value in filter_conditions.items():
                    field_key = key.replace("__", ".")
                    must_conditions.append(
                        models.FieldCondition(
                            key=field_key,
                            match=models.MatchValue(value=value)
                        )
                    )

                query_filter = models.Filter(must=must_conditions)

                search_result = self.client.scroll(
                    collection_name=collection_name,
                    scroll_filter=query_filter,
                    limit=1000,
                    with_payload=False,
                    with_vectors=False
                )

                points = [point.id for point in search_result[0]]

                if not points:
                    return

                self.client.set_payload(
                    collection_name=collection_name,
                    payload=payload_updates,
                    points=points
                )

        await asyncio.to_thread(_update)