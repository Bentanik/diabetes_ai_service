"""
MinIO Configuration - Module cấu hình kết nối MinIO

File này định nghĩa các thông số cấu hình cho kết nối MinIO object storage, bao gồm:
- Endpoint URL
- Thông tin xác thực (access key và secret key)
- Cấu hình bảo mật (SSL)
- Tên các bucket được sử dụng

Tất cả các thông số được lấy từ environment variables với các giá trị mặc định phù hợp.
"""

import os
from typing import Dict, Any
from functools import lru_cache


class MinioConfig:
    """
    Class quản lý cấu hình MinIO.

    Attributes:
        ENDPOINT (str): URL endpoint của MinIO server
        ACCESS_KEY (str): Access key cho xác thực
        SECRET_KEY (str): Secret key cho xác thực
        SECURE (bool): Có sử dụng HTTPS hay không

        # Bucket names
        DOCUMENTS_BUCKET (str): Tên bucket lưu trữ tài liệu
    """

    # Cấu hình kết nối
    ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    ACCESS_KEY: str = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    SECRET_KEY: str = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    SECURE: bool = os.getenv("MINIO_SECURE", "false").lower() == "true"

    # Tên các bucket
    DOCUMENTS_BUCKET: str = os.getenv("MINIO_DOCUMENTS_BUCKET", "documents")

    @classmethod
    @lru_cache(maxsize=1)
    def get_minio_config(cls) -> Dict[str, Any]:
        """
        Lấy cấu hình MinIO dưới dạng dictionary.
        Kết quả được cache để tránh đọc lại environment variables.

        Returns:
            Dict[str, Any]: Dictionary chứa các thông số cấu hình MinIO
        """
        return {
            "endpoint": cls.ENDPOINT,
            "access_key": cls.ACCESS_KEY,
            "secret_key": cls.SECRET_KEY,
            "secure": cls.SECURE,
        }

    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        """
        Kiểm tra tính hợp lệ của cấu hình MinIO.

        Returns:
            Dict[str, Any]: Dictionary chứa kết quả validation với các key:
                - valid (bool): True nếu cấu hình hợp lệ
                - errors (list): Danh sách các lỗi nếu có
        """
        errors = []

        if not cls.ENDPOINT:
            errors.append("MINIO_ENDPOINT không được để trống")

        if not cls.ACCESS_KEY:
            errors.append("MINIO_ACCESS_KEY không được để trống")

        if not cls.SECRET_KEY:
            errors.append("MINIO_SECRET_KEY không được để trống")

        if not cls.DOCUMENTS_BUCKET:
            errors.append("MINIO_DOCUMENTS_BUCKET không được để trống")

        return {"valid": len(errors) == 0, "errors": errors}
