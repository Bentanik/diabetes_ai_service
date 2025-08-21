"""
Process Document Upload Command - Command xử lý tài liệu upload

File này định nghĩa ProcessDocumentUploadCommand để xử lý tài liệu upload.
"""

from dataclasses import dataclass
from core.cqrs import Command


@dataclass
class ProcessDocumentUploadCommand(Command):
    """
    Xử lý tài liệu upload

    Attributes:
        document_job_id (str): ID của DocumentJob
    """

    document_job_id: str
