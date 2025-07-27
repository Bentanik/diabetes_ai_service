import os
from typing import Optional, Dict, Any
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()


class DatabaseConfig:
    """Cấu hình database"""

    # Cấu hình cơ bản
    MONGODB_URL: str = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    DATABASE_NAME: str = os.getenv("DATABASE_NAME", "diabetes_ai")
    MONGODB_USERNAME: Optional[str] = os.getenv("MONGODB_USERNAME")
    MONGODB_PASSWORD: Optional[str] = os.getenv("MONGODB_PASSWORD")

    # Cấu hình connection pool
    MAX_POOL_SIZE: int = int(os.getenv("MONGODB_MAX_POOL_SIZE", "50"))
    MIN_POOL_SIZE: int = int(os.getenv("MONGODB_MIN_POOL_SIZE", "5"))

    # Cấu hình timeout (milliseconds)
    CONNECT_TIMEOUT_MS: int = int(os.getenv("MONGODB_CONNECT_TIMEOUT_MS", "3000"))
    SERVER_SELECTION_TIMEOUT_MS: int = int(
        os.getenv("MONGODB_SERVER_SELECTION_TIMEOUT_MS", "3000")
    )
    SOCKET_TIMEOUT_MS: int = int(os.getenv("MONGODB_SOCKET_TIMEOUT_MS", "20000"))

    # Cấu hình retry
    RETRY_WRITES: bool = os.getenv("MONGODB_RETRY_WRITES", "true").lower() == "true"
    RETRY_READS: bool = os.getenv("MONGODB_RETRY_READS", "true").lower() == "true"

    @classmethod
    @lru_cache(maxsize=1)
    def get_connection_url(cls) -> str:
        """Tạo MongoDB connection URL hoàn chỉnh"""
        url = cls.MONGODB_URL

        if cls.MONGODB_USERNAME and cls.MONGODB_PASSWORD:
            if "://" in url:
                protocol, host_part = url.split("://", 1)
                url = f"{protocol}://{cls.MONGODB_USERNAME}:{cls.MONGODB_PASSWORD}@{host_part}"

        return url

    @classmethod
    @lru_cache(maxsize=1)
    def get_connection_kwargs(cls) -> Dict[str, Any]:
        """Lấy các tham số kết nối MongoDB"""
        return {
            "maxPoolSize": cls.MAX_POOL_SIZE,
            "minPoolSize": cls.MIN_POOL_SIZE,
            "connectTimeoutMS": cls.CONNECT_TIMEOUT_MS,
            "serverSelectionTimeoutMS": cls.SERVER_SELECTION_TIMEOUT_MS,
            "socketTimeoutMS": cls.SOCKET_TIMEOUT_MS,
            "retryWrites": cls.RETRY_WRITES,
            "retryReads": cls.RETRY_READS,
        }

    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        """Kiểm tra tính hợp lệ của cấu hình"""
        errors = []

        if not cls.MONGODB_URL:
            errors.append("MONGODB_URL không được để trống")

        if not cls.DATABASE_NAME:
            errors.append("DATABASE_NAME không được để trống")

        if cls.MIN_POOL_SIZE > cls.MAX_POOL_SIZE:
            errors.append("MIN_POOL_SIZE không thể lớn hơn MAX_POOL_SIZE")

        if cls.CONNECT_TIMEOUT_MS <= 0:
            errors.append("CONNECT_TIMEOUT_MS phải lớn hơn 0")

        return {"valid": len(errors) == 0, "errors": errors}