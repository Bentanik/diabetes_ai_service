"""
Base model chứa các thuộc tính cơ bản của một model

File này định nghĩa BaseModel - một abstract base class cho tất cả các model trong hệ thống.
BaseModel cung cấp các chức năng cơ bản như:
- Quản lý ID (ObjectId từ MongoDB)
- Timestamp tự động (created_at, updated_at)
- Serialization/Deserialization
- Validation và error handling
"""

# Import các thư viện cần thiết
from abc import ABC
from typing import Dict, Union, Optional, TypeVar, Type, Any
from bson import ObjectId
from datetime import datetime
import json

BaseDict = Dict[str, Union[str, int, float, bool, datetime, ObjectId, None]]
T = TypeVar("T", bound="BaseModel")

class BaseModel(ABC):
    """
    Base model chứa các thuộc tính cơ bản của một model
    
    Đây là abstract base class mà tất cả các model khác sẽ kế thừa.
    Cung cấp các chức năng cơ bản như quản lý ID, timestamp, serialization.
    """

    def __init__(self, **kwargs):
        """
        Constructor của BaseModel
        
        Args:
            **kwargs: Các tham số tùy ý sẽ được gán thành thuộc tính của instance
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Nếu không có _id, tạo ObjectId mới
        if not hasattr(self, "_id") or self._id is None:
            self._id = ObjectId()

        current_time = datetime.now()

        # Nếu không có created_at, gán datetime.now()
        # Timestamp khi record được tạo
        if not hasattr(self, "created_at") or self.created_at is None:
            self.created_at = current_time

        # Nếu không có updated_at, gán datetime.now()
        # Timestamp khi record được cập nhật lần cuối
        if not hasattr(self, "updated_at") or self.updated_at is None:
            self.updated_at = current_time

    @property
    def id(self) -> str:
        """
        Property để lấy ID dưới dạng string
        
        Returns:
            str: ID của model dưới dạng string
        """
        return str(self._id)

    @id.setter
    def id(self, value: str):
        """
        Setter cho ID với validation
        
        Args:
            value (str): ID mới dưới dạng string
            
        Raises:
            ValueError: Nếu value không phải string hoặc không phải ObjectId hợp lệ
        """
        # Kiểm tra kiểu dữ liệu
        if not isinstance(value, str):
            raise ValueError("ID phải là một chuỗi")
        
        # Thử chuyển đổi thành ObjectId và validate
        try:
            self._id = ObjectId(value)
        except Exception as e:
            raise ValueError(f"Định dạng ObjectId không hợp lệ: {value}") from e

    def to_dict(self) -> BaseDict:
        """
        Chuyển đổi model thành dictionary với xử lý ObjectId
        
        Returns:
            BaseDict: Dictionary chứa tất cả thuộc tính của model
        """
        result = {
            "_id": self._id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

        return result

    def to_json(self) -> str:
        """
        Chuyển đổi model thành JSON string
        
        Method này sử dụng to_dict() và json.dumps() để tạo JSON string
        từ model. Hữu ích cho API responses hoặc logging.
        
        Returns:
            str: JSON string representation của model
        """
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls: Type[T], data: BaseDict) -> Optional[T]:
        """
        Tạo model từ dictionary (class method)
        
        Class method này cho phép tạo instance mới từ dictionary data.
        Hữu ích khi deserialize từ JSON hoặc database.
        
        Args:
            data (BaseDict): Dictionary chứa dữ liệu để tạo model
            
        Returns:
            Optional[T]: Instance mới của model hoặc None nếu data là None
        """
        if data is None:
            return None
        return cls(**data)

    def update_timestamp(self) -> None:
        """
        Cập nhật timestamp updated_at
        
        Method này được gọi khi model được cập nhật để
        cập nhật thời gian modified.
        """
        self.updated_at = datetime.now()

    def __repr__(self) -> str:
        """
        String representation của model
        
        Returns:
            str: String mô tả model (dùng cho debugging)
        """
        return f"<{self.__class__.__name__}(id={self._id})>"

    def __eq__(self, other: Any) -> bool:
        """
        So sánh hai model bằng ID
        
        Hai model được coi là bằng nhau nếu có cùng ID.
        
        Args:
            other (Any): Object khác để so sánh
            
        Returns:
            bool: True nếu hai model có cùng ID, False nếu không
        """
        if not isinstance(other, BaseModel):
            return False
        return self._id == other._id

    def __hash__(self) -> int:
        """
        Hash dựa trên ID
        
        Hash function cho phép sử dụng model trong set hoặc làm dict key.
        Hash dựa trên _id để đảm bảo tính duy nhất.
        
        Returns:
            int: Hash value của model
        """
        return hash(self._id)
