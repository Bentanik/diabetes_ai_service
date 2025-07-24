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

    @staticmethod
    async def check_duplicate_by_hash(
        documents_collection, file_hash: str
    ) -> Optional[Dict[str, Any]]:
        """
        Kiểm tra file duplicate bằng hash
        Args:
            documents_collection: MongoDB collection
            file_hash: MD5 hash của file
            kb_name: Tên knowledge base (optional)
        Returns:
            Document info nếu trùng, None nếu không trùng
        """
        try:
            query = {"file_hash": file_hash}
            existing_doc = await documents_collection.find_one(query)
            return existing_doc
        except Exception as e:
            logger.error(f"Error checking duplicate: {e}")
            raise

    @staticmethod
    async def check_duplicate_by_name_size(
        documents_collection, file_name: str, file_size: int, kb_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Kiểm tra duplicate bằng tên file + size (backup method)
        """
        try:
            query = {"file_name": file_name, "file_size": file_size}
            if kb_name:
                query["kb_name"] = kb_name

            existing_doc = await documents_collection.find_one(query)
            return existing_doc
        except Exception as e:
            logger.error(f"Error checking duplicate by name/size: {e}")
            raise
