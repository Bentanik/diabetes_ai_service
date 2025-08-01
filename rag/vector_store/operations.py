from typing import Optional, List, Dict, Any
from langchain_core.documents import Document
from rag.vector_store import VectorStoreManager
from qdrant_client.http.models import Filter, FieldCondition, Range, MatchValue
from utils import get_logger
from pydantic import BaseModel
from threading import Lock
from core.llm import get_embedding_model
import asyncio


class SearchResult(BaseModel):
    id: str
    score: float
    payload: Dict[str, Any]
    text: str

    class Config:
        json_encoders = {float: lambda v: float(v)}


def create_qdrant_filter(filter_dict: Optional[Dict[str, Any]]) -> Optional[Filter]:
    """Convert a filter dictionary to a Qdrant Filter object."""
    if not filter_dict:
        return None
    must_conditions = []
    for key, value in filter_dict.items():
        if isinstance(value, dict):
            range_condition = {}
            if "gte" in value:
                range_condition["gte"] = value["gte"]
            if "lte" in value:
                range_condition["lte"] = value["lte"]
            if "gt" in value:
                range_condition["gt"] = value["gt"]
            if "lt" in value:
                range_condition["lt"] = value["lt"]
            if range_condition:
                must_conditions.append(
                    FieldCondition(key=key, range=Range(**range_condition))
                )
        else:
            must_conditions.append(
                FieldCondition(key=key, match=MatchValue(value=value))
            )
    return Filter(must=must_conditions) if must_conditions else None


class VectorStoreOperations:
    _instance: Optional["VectorStoreOperations"] = None
    _lock: Lock = Lock()
    _logger = get_logger(__name__)

    @classmethod
    def get_instance(cls, force_recreate: bool = False) -> "VectorStoreOperations":
        """Get the singleton instance of VectorStoreOperations (thread-safe)."""
        with cls._lock:
            if cls._instance is None:
                cls._logger.info("Khởi tạo VectorStoreOperations singleton")
                cls._instance = cls(force_recreate=force_recreate)
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance (for testing purposes)."""
        with cls._lock:
            cls._instance = None
            cls._logger.info("VectorStoreOperations singleton reset")

    def __init__(self, force_recreate: bool = False):
        """Private constructor to prevent direct instantiation."""
        if VectorStoreOperations._instance is not None:
            raise RuntimeError("Use get_instance() to access VectorStoreOperations")
        self.logger = self._logger
        self.manager = VectorStoreManager(force_recreate=force_recreate)

    async def create_collection(
        self, collection_name: str, vector_size: int = 768
    ) -> None:
        try:
            self.logger.info(f"Đang tạo collection {collection_name}...")
            await self.manager.create_collection_if_not_exists(
                collection_name, vector_size
            )
            self.logger.info(f"Tạo collection {collection_name} thành công trong")
        except Exception as e:
            self.logger.error(
                f"Lỗi tạo collection {collection_name}: {str(e)}", exc_info=True
            )
            raise

    async def store_vectors(
        self,
        texts: List[str],
        collection_name: str,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        vector_size: int = 768,
    ) -> None:
        if not texts:
            raise ValueError("Danh sách texts không được rỗng")
        if metadatas is None:
            metadatas = [{}] * len(texts)
        elif len(metadatas) != len(texts):
            raise ValueError("metadatas phải cùng số lượng với texts")

        for text in texts:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Mỗi text phải là chuỗi không rỗng")

        try:
            self.logger.info(f"Đang lưu {len(texts)} vector vào {collection_name}...")

            vector_store = await self.manager.get_store(collection_name, vector_size)

            documents = [
                Document(page_content=text, metadata=metadata)
                for text, metadata in zip(texts, metadatas)
            ]

            embeddings = await get_embedding_model().embed_documents(
                [doc.page_content for doc in documents]
            )
            self.logger.info(
                f"Generated {len(embeddings)} embeddings, each of size {len(embeddings[0])}"
            )

            vector_store.add_documents(documents=documents)

        except Exception as e:
            self.logger.error(
                f"Lỗi lưu vector vào {collection_name}: {str(e)}", exc_info=True
            )
            raise

    async def search(
        self,
        query_text: str,
        collection_names: Optional[List[str]] = None,
        top_k: int = 5,
        vector_size: int = 768,
        min_score: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Truy vấn vector store, trả về các kết quả có score >= min_score.
        Nếu score là khoảng cách, cần chỉnh lại ngưỡng cho phù hợp.
        """
        try:
            if not query_text.strip():
                raise ValueError("query_text không được rỗng")

            if collection_names is None:
                collection_names = self.get_available_collections()
                if not collection_names:
                    raise ValueError("Không có collection nào khả dụng để tìm kiếm")

            self.logger.info(
                f"Tìm kiếm trên {len(collection_names)} collection(s): {', '.join(collection_names)} với query: {query_text[:50]}..."
            )

            qdrant_filter = create_qdrant_filter(filter)

            all_results = []
            for collection_name in collection_names:
                try:
                    vector_store = await self.manager.get_store(
                        collection_name, vector_size
                    )
                    results = await asyncio.to_thread(
                        vector_store.similarity_search_with_score,
                        query=query_text,
                        k=top_k,
                        filter=qdrant_filter,
                    )
                    formatted_results = [
                        SearchResult(
                            id=doc.metadata.get("id", str(i)),
                            score=float(score),
                            payload={
                                **doc.metadata,
                                "collection_name": collection_name,
                            },
                            text=doc.page_content,
                        )
                        for i, (doc, score) in enumerate(results)
                        if float(score) >= min_score
                    ]
                    all_results.extend(formatted_results)
                except Exception as e:
                    self.logger.warning(
                        f"Lỗi tìm kiếm trong collection {collection_name}: {str(e)}"
                    )
                    continue

            all_results = sorted(all_results, key=lambda x: x.score, reverse=True)[
                :top_k
            ]
            self.logger.info(f"Tìm kiếm hoàn tất")
            return all_results
        except Exception as e:
            self.logger.error(f"Lỗi tìm kiếm: {str(e)}", exc_info=True)
            raise

    def delete_collection(self, collection_name: str) -> None:
        try:
            self.logger.info(f"Đang xóa toàn bộ collection {collection_name}...")
            self.manager.delete_collection(collection_name)
        except Exception as e:
            self.logger.error(
                f"Lỗi xóa collection {collection_name}: {str(e)}", exc_info=True
            )
            raise

    def get_available_collections(self) -> List[str]:
        try:
            return self.manager.get_collection_names()
        except Exception as e:
            self.logger.error(f"Lỗi lấy danh sách collections: {str(e)}", exc_info=True)
            raise
