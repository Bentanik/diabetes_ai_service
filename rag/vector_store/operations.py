from typing import Optional, List, Dict, Any
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
            self.manager.create_collection_if_not_exists(collection_name, vector_size)
        except Exception as e:
            self.logger.error(
                f"Lỗi tạo collection {collection_name}: {str(e)}", exc_info=True
            )
            raise

    def store_vectors(
        self,
        texts: List[str],
        collection_name: str,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        vector_size: int = 768,
    ) -> None:
        # Validate inputs
        if not texts:
            raise ValueError("Danh sách texts không được rỗng")
        if metadatas is None:
            metadatas = [{}] * len(texts)
        elif len(metadatas) != len(texts):
            raise ValueError("metadatas phải cùng số lượng với texts")

        # Validate text content
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Mỗi text phải là chuỗi không rỗng")

        try:
            self.logger.info(f"Đang lưu {len(texts)} vector vào {collection_name}...")

            vector_store = self.manager.get_store(collection_name, vector_size)

            # Create Document objects
            documents = [
                Document(page_content=text, metadata=metadata)
                for text, metadata in zip(texts, metadatas)
            ]

            # Debug embeddings before storing
            from core.llm import get_embedding_model

            embeddings = get_embedding_model().embed_documents(
                [doc.page_content for doc in documents]
            )
            self.logger.info(
                f"Generated {len(embeddings)} embeddings, each of size {len(embeddings[0])}"
            )

            # Store documents
            vector_store.add_documents(documents=documents)

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
            if not query_text.strip():
                raise ValueError("query_text không được rỗng")

            self.logger.info(
                f"Tìm kiếm trong {collection_name} với query: {query_text[:50]}..."
            )

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

            return formatted_results
        except Exception as e:
            self.logger.error(
                f"Lỗi tìm kiếm trong {collection_name}: {str(e)}", exc_info=True
            )
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
