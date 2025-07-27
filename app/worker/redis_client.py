import redis.asyncio as redis
from app.config import RedisConfig

redis_client = redis.Redis(**RedisConfig.get_connection_params())








