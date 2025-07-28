"""
MinIO Manager - Module quản lý tương tác với MinIO storage

File này cung cấp MinioManager class để quản lý các thao tác với MinIO storage như:
- Quản lý buckets
- Upload/download files
- Stream file data
- Xóa files

MinioManager được thiết kế theo pattern Singleton để đảm bảo chỉ có một instance
trong toàn bộ ứng dụng.
"""

from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any
import asyncio
from app.storage.minio_client import MinioClient
from utils import get_logger

logger = get_logger(__name__)


class MinioManager:
    """
    Class quản lý tương tác với MinIO storage.
    Được thiết kế theo pattern Singleton.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        """Khởi tạo các thành phần cần thiết cho MinioManager"""
        logger.info("Khởi tạo MinioManager")
        self.client = MinioClient()
        self.executor = ThreadPoolExecutor(max_workers=5)
        logger.info("MinioManager đã được khởi tạo")

    @classmethod
    def get_instance(cls) -> "MinioManager":
        """
        Lấy instance của MinioManager.

        Returns:
            MinioManager: Instance duy nhất của MinioManager
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def create_bucket_if_not_exists(self, bucket_name: str) -> None:
        """
        Tạo bucket nếu chưa tồn tại.

        Args:
            bucket_name (str): Tên của bucket cần tạo
        """
        if not self.client.bucket_exists(bucket_name):
            logger.info(f"Tạo bucket: {bucket_name}")
            self.client.make_bucket(bucket_name)

    def upload_file(
        self, bucket_name: str, object_name: str, file_data: bytes, content_type: str
    ) -> None:
        """
        Upload file lên MinIO storage.

        Args:
            bucket_name (str): Tên bucket
            object_name (str): Tên object/file
            file_data (bytes): Dữ liệu file
            content_type (str): MIME type của file
        """
        self.create_bucket_if_not_exists(bucket_name)
        logger.info(f"Tải file: {bucket_name}/{object_name}")
        self.client.put_object(
            bucket_name,
            object_name,
            BytesIO(file_data),
            len(file_data),
            content_type,
        )

    def get_file(self, bucket_name: str, object_name: str):
        """
        Lấy file từ MinIO storage.

        Args:
            bucket_name (str): Tên bucket
            object_name (str): Tên object/file

        Returns:
            Response object từ MinIO client
        """
        logger.info(f"Lấy file: {bucket_name}/{object_name}")
        return self.client.get_object(bucket_name, object_name)

    def delete_file(self, bucket_name: str, object_name: str) -> None:
        """
        Xóa file khỏi MinIO storage.

        Args:
            bucket_name (str): Tên bucket
            object_name (str): Tên object/file cần xóa
        """
        logger.info(f"Xóa file: {bucket_name}/{object_name}")
        self.client.remove_object(bucket_name, object_name)

    def get_stream(
        self, bucket_name: str, object_name: str, chunk_size: int = 64 * 1024
    ) -> Dict[str, Any]:
        """
        Lấy stream data của file từ MinIO storage.

        Args:
            bucket_name (str): Tên bucket
            object_name (str): Tên object/file
            chunk_size (int): Kích thước mỗi chunk khi đọc file

        Returns:
            Dict chứa stream generator và metadata của file
        """
        response = self.client.get_object(bucket_name, object_name)
        stat = self.client.stat_object(bucket_name, object_name)

        def stream_generator():
            try:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
            finally:
                response.close()
                response.release_conn()

        return {
            "stream": stream_generator(),
            "size": stat.size,
            "filename": object_name.split("/")[-1],
            "content_type": stat.content_type or "application/octet-stream",
        }

    async def get_stream_async(
        self, bucket_name: str, object_name: str, chunk_size: int = 64 * 1024
    ) -> Dict[str, Any]:
        """
        Phiên bản async của get_stream.

        Args:
            bucket_name (str): Tên bucket
            object_name (str): Tên object/file
            chunk_size (int): Kích thước mỗi chunk khi đọc file

        Returns:
            Dict chứa stream generator và metadata của file
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, self.get_stream, bucket_name, object_name, chunk_size
        )
