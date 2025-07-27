"""
MongoDB Connection - Quản lý kết nối MongoDB với singleton pattern

File này định nghĩa class MongoDBConnection để quản lý kết nối đến MongoDB
database. Sử dụng singleton pattern để đảm bảo chỉ có một instance
kết nối duy nhất trong toàn bộ ứng dụng.

Chức năng chính:
- Singleton pattern để quản lý connection
- Async connection management
- Connection health checking
- Proper error handling và logging
- Connection pooling và resource management
"""

from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from app.config import DatabaseConfig
from utils import get_logger

logger = get_logger(__name__)


class MongoDBConnection:
    """
    Quản lý kết nối MongoDB với singleton pattern
    
    Class này đảm bảo chỉ có một instance kết nối MongoDB duy nhất
    trong toàn bộ ứng dụng, giúp tối ưu resource và tránh connection leaks.
    
    Attributes:
        _instance: Singleton instance
        _client: MongoDB client instance
        _database: Database instance
        _is_connected: Trạng thái kết nối
    """

    _instance: Optional["MongoDBConnection"] = None
    _client: Optional[AsyncIOMotorClient] = None
    _database: Optional[AsyncIOMotorDatabase] = None
    _is_connected: bool = False

    def __new__(cls) -> "MongoDBConnection":
        """
        Singleton pattern - chỉ tạo một instance duy nhất
        
        Returns:
            MongoDBConnection: Singleton instance
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def connect(self) -> None:
        """
        Thiết lập kết nối database
        
        Method này sẽ:
        1. Kiểm tra nếu đã kết nối thì return
        2. Lấy connection URL và parameters từ config
        3. Tạo MongoDB client
        4. Test kết nối bằng ping command
        5. Lấy database instance
        
        Raises:
            ConnectionFailure: Khi không thể kết nối đến MongoDB server
            ServerSelectionTimeoutError: Khi timeout khi chọn server
            Exception: Các lỗi khác trong quá trình kết nối
        """
        # Kiểm tra nếu đã kết nối
        if self._is_connected and self._client:
            logger.info("Database đã được kết nối")
            return

        try:
            logger.info(f"Đang kết nối MongoDB: {DatabaseConfig.DATABASE_NAME}")

            # Lấy URL và tham số kết nối từ config
            connection_url = DatabaseConfig.get_connection_url()
            connection_kwargs = DatabaseConfig.get_connection_kwargs()

            # Tạo MongoDB client với connection pooling
            self._client = AsyncIOMotorClient(connection_url, **connection_kwargs)

            # Test kết nối bằng ping command
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
        """
        Đóng kết nối database
        
        Method này sẽ đóng MongoDB client và reset tất cả state
        về trạng thái ban đầu. Được gọi khi có lỗi hoặc shutdown app.
        """
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
        """
        Kiểm tra kết nối database còn hoạt động không
        
        Method này gửi ping command đến MongoDB để kiểm tra
        xem connection còn hoạt động hay không.
        
        Returns:
            bool: True nếu kết nối còn hoạt động, False nếu không
        """
        if not self._client or not self._is_connected:
            return False

        try:
            await self._client.admin.command("ping")
            return True
        except Exception as e:
            logger.warning(f"Database ping thất bại: {e}")
            return False

    def get_database(self) -> AsyncIOMotorDatabase:
        """
        Lấy database instance
        
        Returns:
            AsyncIOMotorDatabase: Database instance
            
        Raises:
            RuntimeError: Nếu database chưa được kết nối
        """
        if self._database is None or not self._is_connected:
            raise RuntimeError("Database chưa kết nối. Gọi connect() trước.")
        return self._database

    def get_client(self) -> AsyncIOMotorClient:
        """
        Lấy client instance
        
        Returns:
            AsyncIOMotorClient: MongoDB client instance
            
        Raises:
            RuntimeError: Nếu database chưa được kết nối
        """
        if self._client is None or not self._is_connected:
            raise RuntimeError("Database chưa kết nối. Gọi connect() trước.")
        return self._client

    @property
    def is_connected(self) -> bool:
        """
        Kiểm tra trạng thái kết nối
        
        Returns:
            bool: True nếu đã kết nối, False nếu chưa
        """
        return self._is_connected