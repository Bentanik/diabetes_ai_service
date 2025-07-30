from typing import Optional, List, Dict, Any
import time
from langchain_core.documents import Document
from rag.vector_store import VectorStoreManager
from utils import get_logger


class VectorStoreOperations:
    def __init__(self, force_recreate: bool = False):
        self.logger = get_logger(__name__)
        self.manager = VectorStoreManager(force_recreate=force_recreate)

    def create_collection(self, collection_name: str, vector_size: int = 768) -> None:
        try:
            self.logger.info(f"Đang tạo collection {collection_name}...")
            start_time = time.time()
            self.manager.create_collection_if_not_exists(collection_name, vector_size)
            self.logger.info(
                f"Tạo collection {collection_name} thành công trong {time.time() - start_time:.2f}s"
            )
        except Exception as e:
            self.logger.error(
                f"Lỗi tạo collection {collection_name}: {str(e)}", exc_info=True
            )
            raise

    def store_vectors(
        self,
        texts: List[str],
        ids: List[str],
        collection_name: str,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        vector_size: int = 768,
    ) -> None:
        if len(texts) != len(ids):
            raise ValueError("texts và ids phải bằng số lượng")
        if metadatas is None:
            metadatas = [{}] * len(texts)
        elif len(metadatas) != len(texts):
            raise ValueError("metadatas phải cùng số lượng với texts")

        metadatas = [{**md, "id": id_} for md, id_ in zip(metadatas, ids)]

        try:
            self.logger.info(f"Đang lưu {len(texts)} vector vào {collection_name}...")
            start_time = time.time()

            vector_store = self.manager.get_store(collection_name, vector_size)

            documents = [
                Document(page_content=text, metadata=metadata)
                for text, metadata in zip(texts, metadatas)
            ]

            vector_store.add_documents(documents=documents, ids=ids)

            self.logger.info(
                f"Lưu vector thành công trong {time.time() - start_time:.2f}s"
            )
        except Exception as e:
            self.logger.error(
                f"Lỗi lưu vector vào {collection_name}: {str(e)}", exc_info=True
            )
            raise

    def search(
        self,
        query_text: str,
        collection_name: str,
        n_results: int = 5,
        vector_size: int = 768,
    ) -> List[Dict[str, Any]]:
        try:
            self.logger.info(
                f"Tìm kiếm trong {collection_name} với query: {query_text[:50]}..."
            )
            start_time = time.time()

            vector_store = self.manager.get_store(collection_name, vector_size)
            results = vector_store.similarity_search_with_score(
                query=query_text, k=n_results
            )

            formatted_results = [
                {
                    "id": doc.metadata.get("id", str(i)),
                    "score": float(score),
                    "payload": doc.metadata,
                    "text": doc.page_content,
                }
                for i, (doc, score) in enumerate(results)
            ]

            self.logger.info(f"Tìm kiếm hoàn tất trong {time.time() - start_time:.2f}s")
            return formatted_results
        except Exception as e:
            self.logger.error(
                f"Lỗi tìm kiếm trong {collection_name}: {str(e)}", exc_info=True
            )
            raise

    def delete_vectors(
        self, ids: List[str], collection_name: str, vector_size: int = 768
    ) -> None:
        try:
            self.logger.info(f"Đang xóa {len(ids)} vector từ {collection_name}...")
            start_time = time.time()

            vector_store = self.manager.get_store(collection_name, vector_size)
            vector_store.delete(ids=ids)

            self.logger.info(
                f"Xóa vector thành công trong {time.time() - start_time:.2f}s"
            )
        except Exception as e:
            self.logger.error(
                f"Lỗi xóa vector trong {collection_name}: {str(e)}", exc_info=True
            )
            raise

    def get_available_collections(self) -> List[str]:
        return self.manager.get_collection_names()
