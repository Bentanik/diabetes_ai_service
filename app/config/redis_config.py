import os
from typing import Optional

class RedisConfig:
    """Redis configuration settings"""
    
    # Redis connection settings
    HOST: str = os.getenv("REDIS_HOST", "localhost")
    PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    DB: int = int(os.getenv("REDIS_DB", "0"))
    
    # Authentication
    PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD", None)
    
    # Connection pool settings
    MAX_CONNECTIONS: int = int(os.getenv("REDIS_MAX_CONNECTIONS", "10"))
    RETRY_ON_TIMEOUT: bool = os.getenv("REDIS_RETRY_ON_TIMEOUT", "true").lower() == "true"
    
    # Timeout settings
    SOCKET_TIMEOUT: int = int(os.getenv("REDIS_SOCKET_TIMEOUT", "5"))
    SOCKET_CONNECT_TIMEOUT: int = int(os.getenv("REDIS_SOCKET_CONNECT_TIMEOUT", "5"))
    
    # SSL settings
    SSL: bool = os.getenv("REDIS_SSL", "false").lower() == "true"
    SSL_CERT_REQS: Optional[str] = os.getenv("REDIS_SSL_CERT_REQS", None)
    
    @classmethod
    def get_connection_params(cls) -> dict:
        """Get Redis connection parameters as dictionary"""
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
