"""
Database Configuration - Module cấu hình kết nối MongoDB

File này định nghĩa các thông số cấu hình cho kết nối MongoDB, bao gồm:
- Thông số kết nối cơ bản (URL, database name)
- Thông tin xác thực (username, password)
- Cấu hình connection pool
- Timeout settings
- Retry policies

Tất cả các thông số được lấy từ environment variables với các giá trị mặc định phù hợp.
Module sử dụng python-dotenv để load các biến môi trường từ file .env
"""

import os
from typing import Optional, Dict, Any
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()


class DatabaseConfig:
    """
    Class quản lý cấu hình MongoDB.

    Attributes:
        MONGODB_URL (str): URL kết nối đến MongoDB server
        DATABASE_NAME (str): Tên database
        MONGODB_USERNAME (str): Username xác thực (optional)
        MONGODB_PASSWORD (str): Password xác thực (optional)
        MAX_POOL_SIZE (int): Số lượng kết nối tối đa trong pool
        MIN_POOL_SIZE (int): Số lượng kết nối tối thiểu trong pool
        CONNECT_TIMEOUT_MS (int): Thời gian timeout cho kết nối (ms)
        SERVER_SELECTION_TIMEOUT_MS (int): Thời gian timeout cho server selection (ms)
        SOCKET_TIMEOUT_MS (int): Thời gian timeout cho socket operations (ms)
        RETRY_WRITES (bool): Có retry write operations hay không
        RETRY_READS (bool): Có retry read operations hay không
    """

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
        """
        Tạo MongoDB connection URL hoàn chỉnh bao gồm thông tin xác thực.
        Kết quả được cache để tránh xử lý lại.

        Returns:
            str: MongoDB connection URL hoàn chỉnh
        """
        url = cls.MONGODB_URL

        if cls.MONGODB_USERNAME and cls.MONGODB_PASSWORD:
            if "://" in url:
                protocol, host_part = url.split("://", 1)
                url = f"{protocol}://{cls.MONGODB_USERNAME}:{cls.MONGODB_PASSWORD}@{host_part}"

        return url

    @classmethod
    @lru_cache(maxsize=1)
    def get_connection_kwargs(cls) -> Dict[str, Any]:
        """
        Lấy các tham số kết nối MongoDB dưới dạng dictionary.
        Kết quả được cache để tránh xử lý lại.

        Returns:
            Dict[str, Any]: Dictionary chứa các tham số kết nối MongoDB
        """
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
        """
        Kiểm tra tính hợp lệ của cấu hình MongoDB.

        Returns:
            Dict[str, Any]: Dictionary chứa kết quả validation với các key:
                - valid (bool): True nếu cấu hình hợp lệ
                - errors (list): Danh sách các lỗi nếu có
        """
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
