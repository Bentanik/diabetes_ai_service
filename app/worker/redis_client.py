"""
Redis Client - Module kết nối Redis

File này khởi tạo kết nối đến Redis server dựa trên cấu hình từ RedisConfig.
Redis được sử dụng làm message broker cho các background tasks và worker.

Attributes:
    redis_client: Redis client instance được khởi tạo với các thông số kết nối từ config
"""

import redis.asyncio as redis
from app.config import RedisConfig

redis_client = redis.Redis(**RedisConfig.get_connection_params())
