"""
Knowledge Model - Model cho cơ sở tri thức

File này định nghĩa KnowledgeModel - một model kế thừa từ BaseModel để quản lý
cơ sở tri thức trong hệ thống. KnowledgeModel đại diện cho một tập hợp các
tài liệu được sử dụng để huấn luyện AI model.

Chức năng chính:
- Quản lý thông tin cơ sở tri thức (tên, mô tả)
- Theo dõi số lượng và dung lượng tài liệu
- Đánh dấu cơ sở tri thức được chọn để huấn luyện
- Cung cấp các method để thao tác với cơ sở tri thức
"""

from typing import Dict, Union, Optional
from bson import ObjectId
from datetime import datetime

from app.database.models.base_model import BaseModel

KnowledgeDict = Dict[str, Union[str, int, bool, datetime, ObjectId, None]]


class KnowledgeModel(BaseModel):
    """
    Model cho Knowledge (Cơ sở tri thức)
    
    KnowledgeModel đại diện cho một cơ sở tri thức trong hệ thống.
    Mỗi cơ sở tri thức chứa một tập hợp các tài liệu được sử dụng
    để huấn luyện AI model.
    
    Attributes:
        name (str): Tên của cơ sở tri thức
        description (str): Mô tả chi tiết về cơ sở tri thức
        document_count (int): Số lượng tài liệu trong cơ sở tri thức
        total_size_bytes (int): Tổng dung lượng (bytes) của các tài liệu
        select_training (bool): Đánh dấu có được chọn để huấn luyện hay không
    """

    def __init__(
        self,
        name: str,
        description: Optional[str] = "",
        document_count: int = 0,
        total_size_bytes: int = 0,
        select_training: bool = False,
        **kwargs
    ):
        """
        Constructor của KnowledgeModel
        
        Args:
            name (str): Tên của cơ sở tri thức (bắt buộc)
            description (Optional[str]): Mô tả về cơ sở tri thức (mặc định: "")
            document_count (int): Số lượng tài liệu trong cơ sở tri thức (mặc định: 0)
            total_size_bytes (int): Tổng dung lượng (bytes) của các tài liệu (mặc định: 0)
            select_training (bool): Đánh dấu có chọn để huấn luyện hay không (mặc định: False)
            **kwargs: Các tham số bổ sung sẽ được forward đến BaseModel
            
        Raises:
            ValueError: Nếu name rỗng hoặc các giá trị không hợp lệ
        """
        if not name or not name.strip():
            raise ValueError("Tên cơ sở tri thức không được để trống")
        
        if document_count < 0:
            raise ValueError("Số lượng tài liệu không thể âm")
        
        if total_size_bytes < 0:
            raise ValueError("Dung lượng tài liệu không thể âm")
        
        super().__init__(**kwargs)
        
        self.name = name.strip()
        self.description = description or ""
        self.document_count = document_count
        self.total_size_bytes = total_size_bytes
        self.select_training = select_training

    def add_document(self, size_bytes: int) -> None:
        """
        Thêm một tài liệu vào cơ sở tri thức
        
        Method này tăng số lượng tài liệu và dung lượng tổng
        khi có tài liệu mới được thêm vào.
        
        Args:
            size_bytes (int): Dung lượng của tài liệu mới (bytes)
            
        Raises:
            ValueError: Nếu size_bytes âm
        """
        if size_bytes < 0:
            raise ValueError("Dung lượng tài liệu không thể âm")
        
        self.document_count += 1
        self.total_size_bytes += size_bytes
        self.update_timestamp()

    def remove_document(self, size_bytes: int) -> None:
        """
        Xóa một tài liệu khỏi cơ sở tri thức
        
        Method này giảm số lượng tài liệu và dung lượng tổng
        khi có tài liệu bị xóa khỏi cơ sở tri thức.
        
        Args:
            size_bytes (int): Dung lượng của tài liệu bị xóa (bytes)
            
        Raises:
            ValueError: Nếu size_bytes âm hoặc document_count = 0
        """
        if size_bytes < 0:
            raise ValueError("Dung lượng tài liệu không thể âm")
        
        if self.document_count <= 0:
            raise ValueError("Không có tài liệu nào để xóa")
        
        self.document_count -= 1
        self.total_size_bytes -= size_bytes
        self.update_timestamp()

    def get_size_mb(self) -> float:
        """
        Lấy dung lượng cơ sở tri thức theo MB
        
        Returns:
            float: Dung lượng cơ sở tri thức tính bằng MB
        """
        return self.total_size_bytes / (1024 * 1024)

    def get_size_gb(self) -> float:
        """
        Lấy dung lượng cơ sở tri thức theo GB
        
        Returns:
            float: Dung lượng cơ sở tri thức tính bằng GB
        """
        return self.total_size_bytes / (1024 * 1024 * 1024)

    def is_empty(self) -> bool:
        """
        Kiểm tra xem cơ sở tri thức có rỗng không
        
        Returns:
            bool: True nếu không có tài liệu nào, False nếu có
        """
        return self.document_count == 0

    def toggle_training_selection(self) -> None:
        """
        Chuyển đổi trạng thái chọn để huấn luyện
        """
        self.select_training = not self.select_training
        self.update_timestamp()

    def to_dict(self) -> KnowledgeDict:
        """
        Chuyển đổi model thành dictionary

        Returns:
            KnowledgeDict: Dictionary chứa tất cả thuộc tính của KnowledgeModel
        """
        result = super().to_dict()
        
        result.update(
            {
                "_id": self._id,
                "name": self.name,
                "description": self.description,
                "document_count": self.document_count,
                "total_size_bytes": self.total_size_bytes,
                "select_training": self.select_training,
            }
        )
        return result

    @classmethod
    def from_dict(cls, data: KnowledgeDict) -> "KnowledgeModel":
        """
        Tạo KnowledgeModel từ dictionary
        
        Class method này tạo instance mới từ dictionary data.
        Sử dụng pop() để trích xuất các field cần thiết và
        forward các field còn lại cho BaseModel.
        
        Args:
            data (KnowledgeDict): Dictionary chứa dữ liệu để tạo model
            
        Returns:
            KnowledgeModel: Instance mới của KnowledgeModel
            
        Raises:
            ValueError: Nếu data không hợp lệ
        """
        if not data:
            raise ValueError("Dữ liệu không được để trống")
        
        # Tạo copy để tránh modify original data
        data = dict(data)
        
        name = data.pop("name", "")
        description = data.pop("description", "")
        document_count = data.pop("document_count", 0)
        total_size_bytes = data.pop("total_size_bytes", 0)
        select_training = data.pop("select_training", False)
        
        return cls(
            name=name,
            description=description,
            document_count=document_count,
            total_size_bytes=total_size_bytes,
            select_training=select_training,
            **data
        )

    def __repr__(self) -> str:
        """
        String representation của KnowledgeModel
        
        Returns:
            str: String mô tả KnowledgeModel (dùng cho debugging)
        """
        return f"<KnowledgeModel(id={self._id}, name='{self.name}', documents={self.document_count})>"
