"""
Document Model - Module quản lý cơ sở tri thức

File này định nghĩa DocumentModel để lưu trữ thông tin về các tài liệu
được upload hoặc sử dụng để training trong hệ thống.
"""

from typing import Dict, Union, Optional
from bson import ObjectId
from datetime import datetime

from app.database.enums import DocumentType
from app.database.models import BaseModel
from app.database.value_objects import DocumentFile

# Type alias cho dictionary chứa dữ liệu của document
DocumentDict = Dict[str, Union[str, int, bool, datetime, ObjectId, None]]


class DocumentModel(BaseModel):
    """
    Model cho Document (Tài liệu thuộc Knowledge)

    Attributes:
        Thông tin cơ bản:
            knowledge_id (str): ID của cơ sở tri thức chứa tài liệu
            title (str): Tiêu đề của tài liệu
            description (str, optional): Mô tả về tài liệu
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
        file_path: str = "",
        file_size_bytes: int = 0,
        file_hash: Optional[str] = None,
        type: DocumentType = DocumentType.UPLOAD,
        priority_diabetes: float = 0.0,
        **kwargs
    ):
        """
        Khởi tạo một Document mới

        Args:
            Thông tin cơ bản:
                knowledge_id: ID của cơ sở tri thức chứa tài liệu
                title: Tiêu đề tài liệu
                description: Mô tả tài liệu (mặc định: "")
                type: Loại tài liệu (mặc định: UPLOAD)
                priority_diabetes: Độ ưu tiên về tiểu đường (mặc định: 0.0)

            Thông tin file:
                file_path: Đường dẫn file trong storage (mặc định: "")
                file_size_bytes: Kích thước file (mặc định: 0)
                file_hash: Hash của file (mặc định: None)

            **kwargs: Các tham số khác của BaseModel
        """
        super().__init__(**kwargs)
        # Thông tin cơ bản
        self.knowledge_id = knowledge_id
        self.title = title
        self.description = description
        self.type = type
        self.priority_diabetes = priority_diabetes

        # Thông tin file
        self.file = DocumentFile(
            path=file_path,
            size_bytes=file_size_bytes,
            hash=file_hash
        )

    def to_dict(self) -> DocumentDict:
        """
        Chuyển đổi model thành dictionary

        Returns:
            DocumentDict: Dictionary chứa dữ liệu của document
        """
        result = super().to_dict()
        result.update(
            {
                # Thông tin cơ bản
                "knowledge_id": self.knowledge_id,
                "title": self.title,
                "description": self.description,
                "type": self.type,
                "priority_diabetes": self.priority_diabetes,
                
                # Thông tin file
                **self.file.to_dict()
            }
        )
        return result

    @classmethod
    def from_dict(cls, data: DocumentDict) -> "DocumentModel":
        """
        Tạo model từ dictionary

        Args:
            data (DocumentDict): Dictionary chứa dữ liệu document

        Returns:
            DocumentModel: Instance mới của DocumentModel
        """
        data = dict(data)
        
        # Thông tin cơ bản
        knowledge_id = str(data.pop("knowledge_id"))  # Convert ObjectId to str
        title = data.pop("title", "")
        description = data.pop("description", "")
        type = data.pop("type", DocumentType.UPLOAD)
        priority_diabetes = data.pop("priority_diabetes", 0.0)

        # Tạo DocumentFile từ dữ liệu
        file = DocumentFile.from_dict(data)

        return cls(
            knowledge_id=knowledge_id,
            title=title,
            description=description,
            type=type,
            priority_diabetes=priority_diabetes,
            file_path=file.path,
            file_size_bytes=file.size_bytes,
            file_hash=file.hash,
            **data,
        )
