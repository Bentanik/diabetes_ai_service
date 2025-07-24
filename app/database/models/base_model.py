from abc import ABC
from typing import Dict, Union, Optional, TypeVar, Type
from bson import ObjectId
from datetime import datetime

BaseDict = Dict[str, Union[str, int, float, bool, datetime, ObjectId, None]]

T = TypeVar("T", bound="BaseModel")


class BaseModel(ABC):
    """Base model chứa các thuộc tính cơ bản của một model"""

    def __init__(self, **kwargs):
        # Gán tất cả trường trong kwargs thành thuộc tính instance
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Nếu không có _id, tạo ObjectId mới
        if not hasattr(self, "_id") or self._id is None:
            self._id = ObjectId()

        # Nếu không có created_at, gán datetime.now()
        if not hasattr(self, "created_at") or self.created_at is None:
            self.created_at = datetime.now()

        # Nếu không có updated_at, gán datetime.now()
        if not hasattr(self, "updated_at") or self.updated_at is None:
            self.updated_at = datetime.now()

    @property
    def id(self) -> str:
        return str(self._id)

    @id.setter
    def id(self, value: str):
        self._id = ObjectId(value)

    def to_dict(self) -> BaseDict:
        """Convert model thành dictionary"""
        result = {}
        for key, value in self.__dict__.items():
            if not key.startswith("_"):
                result[key] = value
        if self._id:
            result["_id"] = self._id
        return result

    @classmethod
    def from_dict(cls: Type[T], data: BaseDict) -> Optional[T]:
        """Tạo model từ dictionary"""
        if data is None:
            return None
        return cls(**data)

    def update_timestamp(self) -> None:
        """Cập nhật timestamp"""
        self.updated_at = datetime.now()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(id={self._id})>"
