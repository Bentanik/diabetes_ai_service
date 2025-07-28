"""
MinIO Client - Module kết nối và thao tác với MinIO storage

File này cung cấp MinioClient class để tương tác với MinIO object storage service.
MinIO được sử dụng để lưu trữ các tệp tin như tài liệu, hình ảnh, và các binary data khác.
"""

from minio import Minio
from app.config import MinioConfig


class MinioClient:
    """
    Client class để tương tác với MinIO object storage.

    Class này wrap các phương thức cơ bản của Minio SDK để thao tác với buckets và objects:
    - Kiểm tra và tạo bucket
    - Upload/download objects
    - Xem thông tin và xóa objects
    """

    def __init__(self):
        """Khởi tạo MinIO client với thông số từ config"""
        config = MinioConfig.get_minio_config()
        self.client = Minio(
            endpoint=config["endpoint"],
            access_key=config["access_key"],
            secret_key=config["secret_key"],
            secure=config["secure"],
        )

    def bucket_exists(self, bucket_name: str) -> bool:
        """Kiểm tra bucket có tồn tại hay không"""
        return self.client.bucket_exists(bucket_name)

    def make_bucket(self, bucket_name: str):
        """Tạo bucket mới"""
        self.client.make_bucket(bucket_name)

    def put_object(
        self, bucket_name: str, object_name: str, data, length, content_type
    ):
        """Upload object lên bucket"""
        self.client.put_object(bucket_name, object_name, data, length, content_type)

    def get_object(self, bucket_name: str, object_name: str):
        """Download object từ bucket"""
        return self.client.get_object(bucket_name, object_name)

    def stat_object(self, bucket_name: str, object_name: str):
        """Lấy thông tin của object"""
        return self.client.stat_object(bucket_name, object_name)

    def remove_object(self, bucket_name: str, object_name: str):
        """Xóa object khỏi bucket"""
        self.client.remove_object(bucket_name, object_name)
