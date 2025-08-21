"""
Document Job Model - Module quản lý công việc xử lý tài liệu

File này định nghĩa DocumentJobModel để quản lý và theo dõi tiến trình
xử lý các tài liệu trong hệ thống.
"""

from typing import Dict, Any
from app.database.enums import DocumentJobType, DocumentStatus
from app.database.models import BaseModel
from app.database.value_objects import ProcessingStatus, DocumentFile


class DocumentJobModel(BaseModel):
    """
    Model quản lý công việc xử lý tài liệu

    Attributes:
        Thông tin tài liệu:
            document_id (str): ID của tài liệu cần xử lý
            knowledge_id (str): ID của cơ sở tri thức chứa tài liệu
            title (str): Tiêu đề tài liệu
            description (str): Mô tả tài liệu
            file (DocumentFile): Thông tin file tài liệu
            type (DocumentJobType): Loại công việc (upload/training)
            document_status (DocumentStatus): Trạng thái của tài liệu

        Thông tin xử lý:
            processing_status (ProcessingStatus): Trạng thái và tiến độ xử lý

        Thông tin phân loại:
            priority_diabetes (float): Độ ưu tiên về tiểu đường (0.0 - 1.0)
    """

    def __init__(
        self,
        document_id: str,
        knowledge_id: str,
        title: str,
        description: str,
        file: DocumentFile,
        type: DocumentJobType = DocumentJobType.UPLOAD,
        processing_status: ProcessingStatus = ProcessingStatus(),
        priority_diabetes: float = 0.0,
        document_status: DocumentStatus = DocumentStatus.NORMAL,
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
        self.processing_status = processing_status

        # Thông tin phân loại
        self.priority_diabetes = priority_diabetes

        # Trạng thái tài liệu
        self.document_status = document_status

    def to_dict(self) -> Dict[str, Any]:
        """Serialize model to MongoDB dictionary with nested objects preserved."""
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
            "processing_status": (
                self.processing_status.to_dict()
                if hasattr(self.processing_status, "to_dict")
                else self.processing_status
            ),
            # Thông tin file
            "file": (
                self.file.to_dict() if hasattr(self.file, "to_dict") else self.file
            ),
            # Phân loại
            "priority_diabetes": self.priority_diabetes,
            # Trạng thái tài liệu
            "document_status": getattr(self.document_status, "value", self.document_status),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentJobModel":
        if data is None:
            return None

        data = dict(data)

        # Xử lý nested processing_status
        processing_status_data = data.pop("processing_status", None)
        if isinstance(processing_status_data, dict) and processing_status_data:
            processing_status = ProcessingStatus(
                status=processing_status_data.get("status", None),
                progress=processing_status_data.get("progress", 0.0),
                progress_message=processing_status_data.get("progress_message", ""),
            )
        else:
            processing_status = ProcessingStatus(
                status=data.pop("status", None),
                progress=data.pop("progress", 0.0),
                progress_message=data.pop("progress_message", ""),
            )

        # Xử lý nested file với fallback dữ liệu cũ
        file_data = data.pop("file", None)
        if isinstance(file_data, dict) and file_data:
            file = DocumentFile(
                path=file_data.get("file_path", file_data.get("path", "")),
                size_bytes=file_data.get("file_size_bytes", file_data.get("size_bytes", 0)),
                name=file_data.get("file_name", file_data.get("name", "")),
                type=file_data.get("file_type", file_data.get("type", "")),
            )
        else:
            file = DocumentFile(
                path=data.pop("file_path", ""),
                size_bytes=data.pop("file_size_bytes", 0),
                name=data.pop("file_name", ""),
                type=data.pop("file_type", ""),
            )

        # Xử lý document_status enum
        document_status_str = data.pop("document_status", "normal")
        document_status = DocumentStatus(document_status_str)

        return cls(
            document_id=data.pop("document_id", ""),
            knowledge_id=data.pop("knowledge_id", ""),
            title=data.pop("title", ""),
            description=data.pop("description", ""),
            file=file,
            type=data.pop("type", DocumentJobType.UPLOAD),
            processing_status=processing_status,
            priority_diabetes=data.pop("priority_diabetes", 0.0),
            document_status=document_status,
            **data
        )
