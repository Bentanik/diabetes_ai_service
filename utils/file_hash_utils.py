import hashlib
from typing import Optional, Dict, Any
from utils import get_logger

logger = get_logger(__name__)


class FileHashUtils:
    @staticmethod
    def calculate_file_hash(file_path: str, chunk_size: int = 8192) -> str:
        """
        Tính MD5 hash của file
        Args:
            file_path: Đường dẫn file
            chunk_size: Kích thước chunk để đọc (8KB default)
        Returns:
            MD5 hash string (32 ký tự)
        """
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                # Đọc file theo chunk để tiết kiệm memory
                for chunk in iter(lambda: f.read(chunk_size), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {e}")
            raise
