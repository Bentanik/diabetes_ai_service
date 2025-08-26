from qdrant_client import QdrantClient
import os
from typing import Optional

class VectorStoreClient:
    _instance = None

    def __new__(
        cls,
        host: Optional[str] = None,
        port: Optional[int] = None,
        api_key: Optional[str] = None,
        url: Optional[str] = None
    ):
        if cls._instance is not None:
            return cls._instance

        host = host or os.getenv("QDRANT_HOST", "localhost")
        port = port or int(os.getenv("QDRANT_PORT", 6333))
        api_key = api_key or os.getenv("QDRANT_API_KEY")
        url = url or os.getenv("QDRANT_URL")

        try:
            if url:
                client = QdrantClient(
                    url=url,
                    api_key=api_key,
                    timeout=20
                )
                
            else:
                client = QdrantClient(
                    host=host,
                    port=port,
                    api_key=api_key,
                    timeout=20
                )

            client.get_collections()

            cls._instance = super().__new__(cls)
            cls._instance.client = client
            return cls._instance

        except Exception as e:
            raise ConnectionError(f"Không thể kết nối tới Qdrant: {e}")

    @property
    def connection(self) -> QdrantClient:
        if self.client is None:
            raise ValueError("Qdrant client chưa được khởi tạo!")
        return self.client