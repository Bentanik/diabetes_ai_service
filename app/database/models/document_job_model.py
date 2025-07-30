"""
Document Job Model - Module quản lý công việc xử lý tài liệu

File này định nghĩa DocumentJobModel để quản lý và theo dõi tiến trình
xử lý các tài liệu trong hệ thống.
"""

from typing import Dict, Any
from app.database.enums import DocumentJobType
from app.database.models import BaseModel
from app.database.value_objects import ProcessingStatus


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

        Thông tin xử lý:
            processing (ProcessingStatus): Trạng thái và tiến độ xử lý

        Thông tin phân loại:
            is_diabetes (bool): Đánh dấu có liên quan đến tiểu đường
            priority_diabetes (float): Độ ưu tiên về tiểu đường (0.0 - 1.0)
    """

    def __init__(
        self,
        document_id: str,
        knowledge_id: str,
        title: str,
        description: str,
        file_path: str,
        type: DocumentJobType = DocumentJobType.UPLOAD,
        is_diabetes: bool = False,
        status: ProcessingStatus = ProcessingStatus(),
        priority_diabetes: float = 0.0,
        **kwargs
    ):
        """Khởi tạo một công việc xử lý tài liệu mới"""
        super().__init__(**kwargs)
        # Thông tin tài liệu
        self.document_id = document_id
        self.knowledge_id = knowledge_id
        self.title = title
        self.description = description
        self.file_path = file_path
        self.type = type
        self.status = status

        # Thông tin phân loại
        self.is_diabetes = is_diabetes
        self.priority_diabetes = priority_diabetes

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentJobModel":
        """Tạo instance từ MongoDB dictionary"""
        if data is None:
            return None

        # Tạo copy để không modify original data
        data = dict(data)

        # Tạo ProcessingStatus từ dữ liệu
        status = ProcessingStatus(
            status=data.pop("status", None),
            progress=data.pop("progress", 0.0),
            progress_message=data.pop("progress_message", ""),
        )

        return cls(
            document_id=data.pop("document_id", ""),
            knowledge_id=data.pop("knowledge_id", ""),
            title=data.pop("title", ""),
            description=data.pop("description", ""),
            file_path=data.pop("file_path", ""),
            type=data.pop("type", DocumentJobType.UPLOAD),
            is_diabetes=data.pop("is_diabetes", False),
            status=status,
            priority_diabetes=data.pop("priority_diabetes", 0.0),
            **data
        )
