from typing import Dict, Any
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from config import DatabaseConfig
from database import MongoDBConnection
from utils import get_logger

logger = get_logger(__name__)

# Global connection instance
db_connection = MongoDBConnection()


async def connect_to_mongodb() -> None:
    """Kết nối tới MongoDB"""
    await db_connection.connect()


async def close_mongodb_connection() -> None:
    """Đóng kết nối MongoDB"""
    await db_connection.close()


def get_database() -> AsyncIOMotorDatabase:
    """Lấy database instance"""
    return db_connection.get_database()


def get_client() -> AsyncIOMotorClient:
    """Lấy client instance"""
    return db_connection.get_client()


async def check_database_health() -> Dict[str, Any]:
    """Kiểm tra tình trạng database"""
    try:
        is_alive = await db_connection.ping()

        if is_alive:
            return {
                "status": "healthy",
                "connected": True,
                "database": DatabaseConfig.DATABASE_NAME,
            }
        else:
            return {
                "status": "unhealthy",
                "connected": False,
                "error": "Ping thất bại",
            }

    except Exception as e:
        return {
            "status": "error",
            "connected": False,
            "error": str(e),
        }


async def initialize_database() -> None:
    """Khởi tạo database"""
    try:
        logger.info("Đang khởi tạo database...")

        if not db_connection.is_connected:
            await connect_to_mongodb()

        # Kiểm tra kết nối
        health = await check_database_health()
        if health["status"] != "healthy":
            raise RuntimeError(f"Database không khỏe mạnh: {health.get('error')}")

        logger.info("Khởi tạo database thành công")

    except Exception as e:
        logger.error(f"Khởi tạo database thất bại: {e}")
        raise
