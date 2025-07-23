from minio import Minio

from app.config import MinioConfig


class MinioClient:
    def __init__(self):
        config = MinioConfig.get_minio_config()
        self.client = Minio(
            endpoint=config["endpoint"],
            access_key=config["access_key"],
            secret_key=config["secret_key"],
            secure=config["secure"],
        )

    def bucket_exists(self, bucket_name: str) -> bool:
        return self.client.bucket_exists(bucket_name)

    def make_bucket(self, bucket_name: str):
        self.client.make_bucket(bucket_name)

    def put_object(
        self, bucket_name: str, object_name: str, data, length, content_type
    ):
        self.client.put_object(bucket_name, object_name, data, length, content_type)

    def get_object(self, bucket_name: str, object_name: str):
        return self.client.get_object(bucket_name, object_name)

    def stat_object(self, bucket_name: str, object_name: str):
        return self.client.stat_object(bucket_name, object_name)

    def remove_object(self, bucket_name: str, object_name: str):
        self.client.remove_object(bucket_name, object_name)
