"""
Database Manager - Quản lý tổng thể database operations

File này cung cấp interface cao cấp để quản lý database operations,
bao gồm connection management, health checking, và initialization.
Đây là entry point chính để tương tác với database trong ứng dụng.

Chức năng chính:
- Database connection management
- Health checking và monitoring
- Database initialization
- Collections access
- Error handling và logging
"""

from typing import Dict, Any
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from app.config import DatabaseConfig
from app.database.connection import MongoDBConnection
from app.database.collections_config import initialize_collections_and_indexes
from app.database.db_collections import DBCollections
from utils import get_logger

logger = get_logger(__name__)

# Global database connection instance
db_connection = MongoDBConnection()


# ===== Connection Management Functions =====


async def connect_to_mongodb() -> None:
    """
    Kết nối đến MongoDB database

    Function này sẽ thiết lập kết nối đến MongoDB sử dụng
    singleton MongoDBConnection instance.

    Raises:
        ConnectionFailure: Khi không thể kết nối đến MongoDB
        Exception: Các lỗi khác trong quá trình kết nối
    """
    await db_connection.connect()


async def close_mongodb_connection() -> None:
    """
    Đóng kết nối MongoDB

    Function này sẽ đóng kết nối MongoDB và cleanup resources.
    Nên được gọi khi shutdown application.
    """
    await db_connection.close()


# ===== Database Access Functions =====


def get_database() -> AsyncIOMotorDatabase:
    """
    Lấy database instance

    Returns:
        AsyncIOMotorDatabase: Database instance

    Raises:
        RuntimeError: Nếu database chưa được kết nối
    """
    return db_connection.get_database()


def get_client() -> AsyncIOMotorClient:
    """
    Lấy MongoDB client instance

    Returns:
        AsyncIOMotorClient: MongoDB client instance

    Raises:
        RuntimeError: Nếu database chưa được kết nối
    """
    return db_connection.get_client()


def get_collections() -> DBCollections:
    """
    Lấy collections manager

    Returns:
        DBCollections: Collections manager instance

    Raises:
        RuntimeError: Nếu database chưa được kết nối
    """
    return DBCollections(get_database())


# ===== Health Checking Functions =====


async def check_database_health() -> Dict[str, Any]:
    """
    Kiểm tra sức khỏe của database

    Function này sẽ kiểm tra xem database có hoạt động bình thường
    hay không bằng cách ping MongoDB server.

    Returns:
        Dict[str, Any]: Health status với format:
            - status: "healthy" | "unhealthy" | "error"
            - connected: bool
            - database: str (tên database)
            - message: str (thông báo)
            - error: str (thông tin lỗi nếu có)
    """
    try:
        # Ping database để kiểm tra kết nối
        is_alive = await db_connection.ping()

        if is_alive:
            return {
                "status": "healthy",
                "connected": True,
                "database": DatabaseConfig.DATABASE_NAME,
                "message": "Database hoạt động bình thường",
            }
        else:
            return {
                "status": "unhealthy",
                "connected": False,
                "database": DatabaseConfig.DATABASE_NAME,
                "error": "Ping thất bại",
            }
    except Exception as e:
        logger.error(f"Lỗi khi kiểm tra database health: {e}")
        return {
            "status": "error",
            "connected": False,
            "database": DatabaseConfig.DATABASE_NAME,
            "error": str(e),
        }


# ===== Database Initialization Functions =====


async def initialize_database() -> None:
    """
    Khởi tạo database hoàn chỉnh

    Function này sẽ thực hiện các bước sau:
    1. Kết nối đến MongoDB
    2. Kiểm tra health của database
    3. Khởi tạo collections và indexes
    4. Log kết quả khởi tạo

    Raises:
        RuntimeError: Khi database không khỏe mạnh
        Exception: Các lỗi khác trong quá trình khởi tạo
    """
    try:
        logger.info("Đang khởi tạo database...")

        # Bước 1: Kết nối database nếu chưa kết nối
        if not db_connection.is_connected:
            await connect_to_mongodb()

        # Bước 2: Kiểm tra health của database
        health = await check_database_health()
        if health["status"] != "healthy":
            error_msg = health.get("error", "Unknown error")
            raise RuntimeError(f"Database không khỏe mạnh: {error_msg}")

        # Bước 3: Khởi tạo collections và indexes
        await initialize_collections_and_indexes(get_database())

        logger.info("Khởi tạo database thành công")

    except Exception as e:
        logger.error(f"Khởi tạo database thất bại: {e}")
        raise


# ===== Utility Functions =====


def is_database_connected() -> bool:
    """
    Kiểm tra xem database có đang kết nối không

    Returns:
        bool: True nếu đã kết nối, False nếu chưa
    """
    return db_connection.is_connected


async def ensure_database_connection() -> None:
    """
    Đảm bảo database đã được kết nối

    Function này sẽ kết nối database nếu chưa kết nối.
    Hữu ích khi cần đảm bảo connection trước khi thực hiện operations.
    """
    if not db_connection.is_connected:
        await connect_to_mongodb()
