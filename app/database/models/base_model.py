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
from typing import Optional, TypeVar, Type, Dict, Any
from bson import ObjectId
from datetime import datetime
import json

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

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi model thành dictionary để lưu vào MongoDB

        Returns:
            Dict[str, Any]: Dictionary chứa tất cả thuộc tính của model
        """
        result = {}

        # Lấy tất cả attributes của object
        for key, value in self.__dict__.items():
            # Nếu value là một object có method to_dict, gọi nó
            if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
                result.update(value.to_dict())
            # Nếu value là một object có __dict__, flatten nó
            elif hasattr(value, "__dict__") and not isinstance(
                value, (str, int, float, bool, datetime, ObjectId, type(None))
            ):
                result.update(value.__dict__)
            else:
                result[key] = value

        return result

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> Optional[T]:
        """
        Tạo model từ dictionary (từ MongoDB)

        Args:
            data (Dict[str, Any]): Dictionary chứa dữ liệu từ MongoDB

        Returns:
            Optional[T]: Instance mới của model hoặc None nếu data là None
        """
        if data is None:
            return None

        # Tạo instance với tất cả data từ MongoDB
        # Các model con sẽ override method này để xử lý logic riêng
        return cls(**data)

    def to_json(self) -> str:
        """
        Chuyển đổi model thành JSON string

        Returns:
            str: JSON string representation của model
        """
        return json.dumps(
            {
                "_id": str(self._id),
                "created_at": self.created_at.isoformat(),
                "updated_at": self.updated_at.isoformat(),
            }
        )

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

    def __eq__(self, other: object) -> bool:
        """
        So sánh hai model bằng ID

        Hai model được coi là bằng nhau nếu có cùng ID.

        Args:
            other (object): Object khác để so sánh

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
