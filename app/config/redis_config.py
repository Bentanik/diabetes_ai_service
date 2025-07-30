"""
Redis Configuration - Module cấu hình kết nối Redis

File này định nghĩa các thông số cấu hình cho kết nối Redis, bao gồm:
- Thông số kết nối cơ bản (host, port, database)
- Xác thực (password)
- Cấu hình connection pool
- Timeout và retry settings
- SSL configuration

Tất cả các thông số được lấy từ environment variables với các giá trị mặc định phù hợp.
"""

import os
from typing import Optional


class RedisConfig:
    """
    Class quản lý cấu hình Redis.

    Attributes:
        HOST (str): Redis server host
        PORT (int): Redis server port
        DB (int): Redis database index
        PASSWORD (str): Mật khẩu xác thực Redis (optional)
        MAX_CONNECTIONS (int): Số lượng kết nối tối đa trong pool
        RETRY_ON_TIMEOUT (bool): Có retry khi timeout hay không
        SOCKET_TIMEOUT (int): Thời gian timeout cho socket operations
        SOCKET_CONNECT_TIMEOUT (int): Thời gian timeout cho kết nối
        SSL (bool): Có sử dụng SSL hay không
        SSL_CERT_REQS (str): Yêu cầu về SSL certificate
    """

    # Redis connection settings
    HOST: str = os.getenv("REDIS_HOST", "localhost")
    PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    DB: int = int(os.getenv("REDIS_DB", "0"))

    # Authentication
    PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD", None)

    # Connection pool settings
    MAX_CONNECTIONS: int = int(os.getenv("REDIS_MAX_CONNECTIONS", "10"))
    RETRY_ON_TIMEOUT: bool = (
        os.getenv("REDIS_RETRY_ON_TIMEOUT", "true").lower() == "true"
    )

    # Timeout settings
    SOCKET_TIMEOUT: int = int(
        os.getenv("REDIS_SOCKET_TIMEOUT", "30")
    )  # Tăng từ 5 lên 30
    SOCKET_CONNECT_TIMEOUT: int = int(
        os.getenv("REDIS_SOCKET_CONNECT_TIMEOUT", "10")
    )  # Tăng từ 5 lên 10

    # SSL settings
    SSL: bool = os.getenv("REDIS_SSL", "false").lower() == "true"
    SSL_CERT_REQS: Optional[str] = os.getenv("REDIS_SSL_CERT_REQS", None)

    @classmethod
    def get_connection_params(cls) -> dict:
        """
        Lấy các thông số kết nối Redis dưới dạng dictionary.

        Returns:
            dict: Dictionary chứa các thông số kết nối Redis được cấu hình
        """
        params = {
            "host": cls.HOST,
            "port": cls.PORT,
            "db": cls.DB,
            "max_connections": cls.MAX_CONNECTIONS,
            "retry_on_timeout": cls.RETRY_ON_TIMEOUT,
            "socket_timeout": cls.SOCKET_TIMEOUT,
            "socket_connect_timeout": cls.SOCKET_CONNECT_TIMEOUT,
        }

        if cls.PASSWORD:
            params["password"] = cls.PASSWORD

        if cls.SSL:
            params["ssl"] = cls.SSL
            if cls.SSL_CERT_REQS:
                params["ssl_cert_reqs"] = cls.SSL_CERT_REQS

        return params
