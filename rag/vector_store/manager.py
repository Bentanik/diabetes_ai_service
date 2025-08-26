from typing import List, Dict, Optional
from qdrant_client.http import models
import uuid
import asyncio
import logging
from .client import VectorStoreClient

logger = logging.getLogger(__name__)

class VectorStoreManager:

    def __init__(self):
        self.client = VectorStoreClient().connection

    async def create_collection_async(
        self,
        name: str,
        size: int = 768,
        distance: str = "Cosine"
    ) -> bool:
        """
        Tạo collection nếu chưa tồn tại.
        :return: True nếu tạo mới, False nếu đã tồn tại.
        """
        def _create():
            try:
                collections = self.client.get_collections().collections
                if name in [c.name for c in collections]:
                    return False
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=models.VectorParams(size=size, distance=distance)
                )
                logger.info(f"Created collection '{name}' with {distance} on {size}D vectors.")
                return True
            except Exception as e:
                logger.error(f"Failed to create collection '{name}': {e}")
                raise

        return await asyncio.to_thread(_create)

    async def delete_collection_async(self, name: str) -> None:
        """Xóa collection (nếu tồn tại)."""
        def _delete():
            try:
                self.client.delete_collection(collection_name=name)
                logger.info(f"Deleted collection '{name}'.")
            except Exception as e:
                logger.warning(f"Failed to delete collection '{name}': {e}")

        await asyncio.to_thread(_delete)

    async def insert_async(
        self,
        name: str,
        embeddings: List[List[float]],
        payloads: Optional[List[Dict]] = None
    ) -> List[str]:
        """
        Chèn danh sách embedding vào collection.
        :return: Danh sách ID đã tạo.
        """
        if not embeddings:
            return []

        def _insert():
            ids = [str(uuid.uuid4()) for _ in embeddings]
            final_payloads = payloads or [{} for _ in embeddings]
            try:
                self.client.upsert(
                    collection_name=name,
                    points=models.Batch(
                        ids=ids,
                        vectors=embeddings,
                        payloads=final_payloads
                    )
                )
                logger.debug(f"Inserted {len(ids)} points into '{name}'.")
                return ids
            except Exception as e:
                logger.error(f"Insert failed in '{name}': {e}")
                raise

        return await asyncio.to_thread(_insert)

    async def search_async(
        self,
        collections: List[str],
        query_vector: List[float],
        top_k: int = 5,
        score_threshold: float = 0.7,
        **filters
    ) -> Dict[str, List[Dict]]:
        """
        Tìm kiếm trong nhiều collections.
        :param collections: Danh sách tên collection.
        :param query_vector: Vector truy vấn.
        :param top_k: Số lượng kết quả mỗi collection.
        :param score_threshold: Ngưỡng điểm (0~1), càng cao càng chính xác.
        :param filters: Điều kiện lọc, ví dụ: user_id="user_123", metadata__doc_type="manual"
        :return: Dict với key là tên collection, value là danh sách kết quả.
        """
        if not collections:
            raise ValueError("collections không được rỗng.")
        if top_k <= 0:
            raise ValueError("top_k phải > 0.")
        if len(query_vector) == 0:
            raise ValueError("query_vector không được rỗng.")

        def _search():
            results = {}
            for col in collections:
                try:
                    # Xây dựng filter
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

                    hits = self.client.search(
                        collection_name=col,
                        query_vector=query_vector,
                        limit=top_k,
                        score_threshold=score_threshold,
                        query_filter=query_filter,
                        with_payload=True,
                        with_vectors=False
                    )

                    results[col] = [
                        {
                            "id": hit.id,
                            "payload": hit.payload,
                            "score": hit.score
                        }
                        for hit in hits
                    ]
                except Exception as e:
                    logger.warning(f"Search failed on collection '{col}': {e}")
                    results[col] = []  # Trả về mảng rỗng thay vì lỗi
            return results

        return await asyncio.to_thread(_search)

    async def delete_by_metadata_async(self, collection_name: str, **conditions) -> None:
        """
        Xóa các điểm (points) theo điều kiện metadata.
        Ví dụ: await manager.delete_by_metadata_async("docs", user_id="u123", doc_id="d456")
        """
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

            try:
                self.client.delete(
                    collection_name=collection_name,
                    points_selector=models.FilterSelector(filter=query_filter)
                )
                logger.info(f"Deleted points from '{collection_name}' with {conditions}")
            except Exception as e:
                logger.error(f"Delete failed in '{collection_name}': {e}")
                raise

        await asyncio.to_thread(_delete)

    async def update_payload_async(
        self,
        collection_name: str,
        payload_updates: Dict,
        point_id: Optional[str] = None,
        **filter_conditions
    ) -> None:
        """
        Cập nhật payload cho một điểm hoặc nhiều điểm qua filter.
        - Nếu có point_id: cập nhật điểm đó.
        - Nếu có filter_conditions: cập nhật tất cả điểm khớp.
        """
        if not point_id and not filter_conditions:
            raise ValueError("Cần point_id hoặc ít nhất một điều kiện filter.")

        def _update():
            try:
                if point_id:
                    self.client.set_payload(
                        collection_name=collection_name,
                        payload=payload_updates,
                        points=[point_id]
                    )
                    logger.debug(f"Updated point {point_id} in '{collection_name}'")
                else:
                    # Tìm tất cả điểm khớp filter
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

                    all_points = []
                    offset = None
                    while True:
                        result, next_offset = self.client.scroll(
                            collection_name=collection_name,
                            scroll_filter=query_filter,
                            limit=100,
                            with_payload=False,
                            with_vectors=False,
                            offset=offset
                        )
                        all_points.extend([point.id for point in result])
                        if not next_offset:
                            break
                        offset = next_offset

                    if not all_points:
                        logger.debug(f"No points matched filter {filter_conditions}")
                        return

                    self.client.set_payload(
                        collection_name=collection_name,
                        payload=payload_updates,
                        points=all_points
                    )
                    logger.info(f"Updated {len(all_points)} points in '{collection_name}'")
            except Exception as e:
                logger.error(f"Update payload failed: {e}")
                raise

        await asyncio.to_thread(_update)