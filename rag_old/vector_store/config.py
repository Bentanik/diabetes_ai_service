from typing import Dict, Any
from functools import lru_cache


class VectorStoreConfig:
    """
    Class quản lý cấu hình Qdrant vector store, hỗ trợ nhiều collection tự tạo động.
    """

    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_API_KEY: str = None
    QDRANT_PREFER_GRPC: bool = False
    QDRANT_REQUEST_TIMEOUT: float = 5.0

    COLLECTIONS: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def add_collection(
        cls, collection_name: str, vector_size: int, distance: str
    ) -> None:
        if not collection_name:
            raise ValueError("Tên collection không được để trống")
        if vector_size <= 0:
            raise ValueError("vector_size phải lớn hơn 0")
        if distance.upper() not in ["COSINE", "EUCLID", "DOT"]:
            raise ValueError("distance phải là một trong: COSINE, EUCLID, DOT")

        cls.COLLECTIONS[collection_name] = {
            "vector_size": vector_size,
            "distance": distance.upper(),
        }

    @classmethod
    @lru_cache(maxsize=1)
    def get_connection_config(cls) -> Dict[str, Any]:
        config = {
            "host": cls.QDRANT_HOST,
            "port": cls.QDRANT_PORT,
            "api_key": cls.QDRANT_API_KEY,
            "prefer_grpc": cls.QDRANT_PREFER_GRPC,
            "timeout": cls.QDRANT_REQUEST_TIMEOUT,
        }
        return {k: v for k, v in config.items() if v is not None}

    @classmethod
    @lru_cache(maxsize=1)
    def get_collection_configs(cls) -> Dict[str, Dict[str, Any]]:
        return cls.COLLECTIONS

    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        errors = []

        if not cls.QDRANT_HOST:
            errors.append("QDRANT_HOST không được để trống")

        if not cls.QDRANT_PORT or cls.QDRANT_PORT <= 0:
            errors.append("QDRANT_PORT phải là số nguyên dương")

        if not cls.COLLECTIONS:
            errors.append(
                "Phải có ít nhất một collection được khai báo bằng add_collection()"
            )

        for name, config in cls.COLLECTIONS.items():
            if not name:
                errors.append("Tên collection không được để trống")
            if config["vector_size"] <= 0:
                errors.append(f"{name}: vector_size phải lớn hơn 0")
            if config["distance"] not in ["COSINE", "EUCLID", "DOT"]:
                errors.append(
                    f"{name}: distance phải là một trong: COSINE, EUCLID, DOT"
                )

        if cls.QDRANT_REQUEST_TIMEOUT <= 0:
            errors.append("QDRANT_REQUEST_TIMEOUT phải lớn hơn 0")

        return {"valid": len(errors) == 0, "errors": errors}
