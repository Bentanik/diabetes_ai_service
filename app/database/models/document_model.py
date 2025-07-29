"""
Document Model - Module quản lý cơ sở tri thức

File này định nghĩa DocumentModel để lưu trữ thông tin về các tài liệu
được upload hoặc sử dụng để training trong hệ thống.
"""

from typing import Optional, Dict, Any

from app.database.enums import DocumentType
from app.database.models import BaseModel
from app.database.value_objects import DocumentFile


class DocumentModel(BaseModel):
    """
    Model cho Document (Tài liệu thuộc Knowledge)

    Attributes:
        Thông tin cơ bản:
            knowledge_id (str): ID của cơ sở tri thức chứa tài liệu
            title (str): Tiêu đề của tài liệu
            description (str): Mô tả về tài liệu
            type (DocumentType): Loại tài liệu (upload hoặc training)
            priority_diabetes (float): Độ ưu tiên liên quan đến bệnh tiểu đường (0.0-1.0)

        Thông tin file:
            file (DocumentFile): Đối tượng chứa thông tin về file
    """

    def __init__(
        self,
        knowledge_id: str,
        title: str,
        description: Optional[str] = "",
        type: DocumentType = DocumentType.UPLOAD,
        priority_diabetes: float = 0.0,
        file: Optional[DocumentFile] = None,
        **kwargs
    ):
        """Khởi tạo một Document mới"""
        super().__init__(**kwargs)
        # Thông tin cơ bản
        self.knowledge_id = knowledge_id
        self.title = title
        self.description = description
        self.type = type
        self.priority_diabetes = priority_diabetes

        # Thông tin file
        self.file = file or DocumentFile()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentModel":
        """Tạo instance từ MongoDB dictionary"""
        if data is None:
            return None

        # Tạo copy để không modify original data
        data = dict(data)

        # Thông tin cơ bản
        knowledge_id = str(data.pop("knowledge_id", ""))
        title = data.pop("title", "")
        description = data.pop("description", "")
        type = data.pop("type", DocumentType.UPLOAD)
        priority_diabetes = data.pop("priority_diabetes", 0.0)

        # Tạo DocumentFile từ dữ liệu
        file = DocumentFile(
            path=data.pop("file_path", ""),
            size_bytes=data.pop("file_size_bytes", 0),
            hash=data.pop("file_hash", None),
        )

        return cls(
            knowledge_id=knowledge_id,
            title=title,
            description=description,
            type=type,
            priority_diabetes=priority_diabetes,
            file=file,
            **data
        )
