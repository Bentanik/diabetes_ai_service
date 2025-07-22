from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from app.config import DatabaseConfig
from utils import get_logger

logger = get_logger(__name__)


class MongoDBConnection:
    """Quản lý kết nối MongoDB với singleton pattern"""

    _instance: Optional["MongoDBConnection"] = None
    _client: Optional[AsyncIOMotorClient] = None
    _database: Optional[AsyncIOMotorDatabase] = None
    _is_connected: bool = False

    def __new__(cls) -> "MongoDBConnection":
        """Singleton pattern - chỉ tạo một instance duy nhất"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def connect(self) -> None:
        """Thiết lập kết nối database"""
        if self._is_connected and self._client:
            logger.info("Database đã được kết nối")
            return

        try:
            logger.info(f"Đang kết nối MongoDB: {DatabaseConfig.DATABASE_NAME}")

            # Lấy URL và tham số kết nối
            connection_url = DatabaseConfig.get_connection_url()
            connection_kwargs = DatabaseConfig.get_connection_kwargs()

            # Tạo client
            self._client = AsyncIOMotorClient(connection_url, **connection_kwargs)

            # Test kết nối
            await self._client.admin.command("ping")

            # Lấy database instance
            self._database = self._client[DatabaseConfig.DATABASE_NAME]
            self._is_connected = True

            logger.info(f"Kết nối MongoDB thành công: {DatabaseConfig.DATABASE_NAME}")

        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"Lỗi kết nối MongoDB: {e}")
            await self.close()
            raise
        except Exception as e:
            logger.error(f"Lỗi không mong muốn khi kết nối MongoDB: {e}")
            await self.close()
            raise

    async def close(self) -> None:
        """Đóng kết nối database"""
        if self._client:
            try:
                logger.info("Đang đóng kết nối MongoDB...")
                self._client.close()
                self._client = None
                self._database = None
                self._is_connected = False
                logger.info("Đóng kết nối MongoDB thành công")
            except Exception as e:
                logger.error(f"Lỗi khi đóng kết nối MongoDB: {e}")

    async def ping(self) -> bool:
        """Kiểm tra kết nối database còn hoạt động không"""
        if not self._client or not self._is_connected:
            return False

        try:
            await self._client.admin.command("ping")
            return True
        except Exception as e:
            logger.warning(f"Database ping thất bại: {e}")
            return False

    def get_database(self) -> AsyncIOMotorDatabase:
        """Lấy database instance"""
        if self._database is None or not self._is_connected:
            raise RuntimeError("Database chưa kết nối. Gọi connect() trước.")
        return self._database

    def get_client(self) -> AsyncIOMotorClient:
        """Lấy client instance"""
        if self._database is None or not self._is_connected:
            raise RuntimeError("Database chưa kết nối. Gọi connect() trước.")
        return self._client

    @property
    def is_connected(self) -> bool:
        """Kiểm tra trạng thái kết nối"""
        return self._is_connected
