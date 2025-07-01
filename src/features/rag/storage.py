"""MinIO storage integration for document storage."""

import os
from datetime import datetime
from typing import Dict, Any, Optional

from core.minio_client import minio_client
from core.logging_config import get_logger

logger = get_logger(__name__)


class DocumentStorage:
    """Document storage using MinIO."""

    def __init__(self):
        self.client = minio_client

    def store_document(
        self,
        file_data: bytes,
        filename: str,
        knowledge_name: str,
        content_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Store a document in MinIO.

        Args:
            file_data: The file content in bytes
            filename: Original filename
            knowledge_name: Knowledge base name (used as bucket name)
            content_type: File content type
            metadata: Optional metadata

        Returns:
            Dict with storage information
        """
        try:
            # Generate a unique object name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            object_name = f"{timestamp}_{filename}"

            # Store the file
            self.client.upload_file(
                bucket_name=knowledge_name,
                object_name=object_name,
                file_data=file_data,
                content_type=content_type,
            )

            # Return storage info
            return {
                "storage_path": f"{knowledge_name}/{object_name}",
                "storage_time": datetime.now().isoformat(),
                "size_bytes": len(file_data),
                "content_type": content_type,
                "metadata": metadata or {},
            }

        except Exception as e:
            logger.error(f"Error storing document in MinIO: {str(e)}")
            raise Exception(f"Failed to store document: {str(e)}")

    def get_document(self, knowledge_name: str, object_name: str):
        """
        Get a document from MinIO.

        Args:
            knowledge_name: Knowledge base name (bucket name)
            object_name: Object name in the bucket

        Returns:
            The document data
        """
        try:
            return self.client.get_file(knowledge_name, object_name)
        except Exception as e:
            logger.error(f"Error getting document from MinIO: {str(e)}")
            raise Exception(f"Failed to get document: {str(e)}")


document_storage = DocumentStorage()
