from qdrant_client import QdrantClient

class VectorStoreClient:
    _instance = None

    def __new__(cls, host="localhost", port=6333):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.client = QdrantClient(host=host, port=port)
        return cls._instance

    @property
    def connection(self):
        return self.client
