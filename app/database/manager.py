# app/database.py

from typing import Dict, Any
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from app.config import DatabaseConfig
from app.database import (
    MongoDBConnection,
    initialize_collections_and_indexes,
    DBCollections,
)
from utils import get_logger

logger = get_logger(__name__)

COLLECTION_INDEX_CONFIG = {
    "knowledges": [
        {"fields": [("name", 1)], "unique": True, "name": "name_unique_idx"},
    ],
}


db_connection = MongoDBConnection()


# ---- Helper functions ----
async def connect_to_mongodb() -> None:
    await db_connection.connect()


async def close_mongodb_connection() -> None:
    await db_connection.close()


def get_database() -> AsyncIOMotorDatabase:
    return db_connection.get_database()


def get_client() -> AsyncIOMotorClient:
    return db_connection.get_client()


def get_collections() -> DBCollections:
    return DBCollections(get_database())


async def check_database_health() -> Dict[str, Any]:
    try:
        is_alive = await db_connection.ping()
        if is_alive:
            return {
                "status": "healthy",
                "connected": True,
                "database": DatabaseConfig.DATABASE_NAME,
            }
        else:
            return {"status": "unhealthy", "connected": False, "error": "Ping thất bại"}
    except Exception as e:
        return {"status": "error", "connected": False, "error": str(e)}


# ---- Hàm khởi tạo database tổng ----
async def initialize_database() -> None:
    try:
        logger.info("Đang khởi tạo database...")

        if not db_connection.is_connected:
            await connect_to_mongodb()

        health = await check_database_health()
        if health["status"] != "healthy":
            raise RuntimeError(f"Database không khỏe mạnh: {health.get('error')}")

        await initialize_collections_and_indexes(get_database())

        logger.info("Khởi tạo database thành công")
    except Exception as e:
        logger.error(f"Khởi tạo database thất bại: {e}")
        raise
