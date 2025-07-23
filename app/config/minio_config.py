import os
from typing import Dict, Any
from functools import lru_cache


class MinioConfig:
    """Cấu hình Minio"""

    ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    ACCESS_KEY: str = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    SECRET_KEY: str = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    SECURE: bool = os.getenv("MINIO_SECURE", "false").lower() == "true"

    @classmethod
    @lru_cache(maxsize=1)
    def get_minio_config(cls) -> Dict[str, Any]:
        """Lấy config Minio"""
        return {
            "endpoint": cls.ENDPOINT,
            "access_key": cls.ACCESS_KEY,
            "secret_key": cls.SECRET_KEY,
            "secure": cls.SECURE,
        }

    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        errors = []

        if not cls.ENDPOINT:
            errors.append("MINIO_ENDPOINT không được để trống")

        if not cls.ACCESS_KEY:
            errors.append("MINIO_ACCESS_KEY không được để trống")

        if not cls.SECRET_KEY:
            errors.append("MINIO_SECRET_KEY không được để trống")

        return {"valid": len(errors) == 0, "errors": errors}
