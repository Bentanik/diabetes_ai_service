"""
Document Job Model - Module quản lý công việc xử lý tài liệu

File này định nghĩa DocumentJobModel để quản lý và theo dõi tiến trình
xử lý các tài liệu trong hệ thống.
"""

from typing import Dict, Any
from app.database.enums import DocumentJobType
from app.database.models import BaseModel
from app.database.value_objects import ProcessingStatus
from app.database.value_objects.document_job_file import DocumentJobFile


class DocumentJobModel(BaseModel):
    """
    Model quản lý công việc xử lý tài liệu

    Attributes:
        Thông tin tài liệu:
            document_id (str): ID của tài liệu cần xử lý
            knowledge_id (str): ID của cơ sở tri thức chứa tài liệu
            title (str): Tiêu đề tài liệu
            description (str): Mô tả tài liệu
            file_path (str): Đường dẫn đến file
            type (DocumentJobType): Loại công việc (upload/training)
            is_document_delete (bool): Có xóa tài liệu gốc chưa

        Thông tin xử lý:
            processing (ProcessingStatus): Trạng thái và tiến độ xử lý

        Thông tin phân loại:
            priority_diabetes (float): Độ ưu tiên về tiểu đường (0.0 - 1.0)
    """

    def __init__(
        self,
        document_id: str,
        knowledge_id: str,
        title: str,
        description: str,
        file: DocumentJobFile,
        type: DocumentJobType = DocumentJobType.UPLOAD,
        status: ProcessingStatus = ProcessingStatus(),
        priority_diabetes: float = 0.0,
        is_document_delete: bool = False,
        is_document_duplicate: bool = False,
        **kwargs
    ):
        """Khởi tạo một công việc xử lý tài liệu mới"""
        super().__init__(**kwargs)
        # Thông tin tài liệu
        self.document_id = document_id
        self.knowledge_id = knowledge_id
        self.title = title
        self.description = description
        self.file = file
        self.type = type
        self.status = status

        # Thông tin phân loại
        self.priority_diabetes = priority_diabetes

        self.is_document_delete = is_document_delete
        self.is_document_duplicate = is_document_duplicate

    def to_dict(self) -> Dict[str, Any]:
        """Serialize model to MongoDB dictionary with nested objects preserved."""
        # Lưu ý: Không dùng BaseModel.to_dict để tránh flatten nested objects
        return {
            "_id": getattr(self, "_id", None),
            "created_at": getattr(self, "created_at", None),
            "updated_at": getattr(self, "updated_at", None),
            # Thông tin tài liệu
            "document_id": self.document_id,
            "knowledge_id": self.knowledge_id,
            "title": self.title,
            "description": self.description,
            "type": getattr(self.type, "value", self.type),
            # Thông tin xử lý
            "status": (
                self.status.to_dict()
                if hasattr(self.status, "to_dict")
                else self.status
            ),
            # Thông tin file
            "file": (
                self.file.to_dict() if hasattr(self.file, "to_dict") else self.file
            ),
            # Phân loại
            "priority_diabetes": self.priority_diabetes,
            "is_document_delete": self.is_document_delete,
            "is_document_duplicate": self.is_document_duplicate,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentJobModel":
        if data is None:
            return None

        data = dict(data)

        # Ưu tiên đọc nested status, fallback về các field flat nếu có
        status_data = data.pop("status", None)
        if isinstance(status_data, dict) and status_data:
            status = ProcessingStatus(
                status=status_data.get("status", None),
                progress=status_data.get("progress", 0.0),
                progress_message=status_data.get("progress_message", ""),
            )
        else:
            status = ProcessingStatus(
                status=data.pop("status", None),
                progress=data.pop("progress", 0.0),
                progress_message=data.pop("progress_message", ""),
            )

        # Ưu tiên đọc nested file, fallback về các field flat nếu có
        file_data = data.pop("file", None)
        if isinstance(file_data, dict) and file_data:
            file = DocumentJobFile(
                path=file_data.get("file_path", ""),
                size_bytes=file_data.get("file_size_bytes", 0),
                file_name=file_data.get("file_name", ""),
                file_type=file_data.get("file_type", ""),
            )
        else:
            file = DocumentJobFile(
                path=data.pop("file_path", ""),
                size_bytes=data.pop("file_size_bytes", 0),
                file_name=data.pop("file_name", ""),
                file_type=data.pop("file_type", ""),
            )

        return cls(
            document_id=data.pop("document_id", ""),
            knowledge_id=data.pop("knowledge_id", ""),
            title=data.pop("title", ""),
            description=data.pop("description", ""),
            file=file,
            type=data.pop("type", DocumentJobType.UPLOAD),
            status=status,
            priority_diabetes=data.pop("priority_diabetes", 0.0),
            is_document_delete=data.pop("is_document_delete", False),
            is_document_duplicate=data.pop("is_document_duplicate", False),
            **data
        )
