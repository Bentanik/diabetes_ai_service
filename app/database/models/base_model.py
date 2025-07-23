from abc import ABC
from typing import Dict, Union, Optional, TypeVar, Type
from bson import ObjectId
from datetime import datetime

BaseDict = Dict[str, Union[str, int, float, bool, datetime, ObjectId, None]]

T = TypeVar("T", bound="BaseModel")


class BaseModel(ABC):
    """Base model chứa các thuộc tính cơ bản của một model"""

    def __init__(self, **kwargs):
        self._id: ObjectId = kwargs.get("_id") or ObjectId()
        self.created_at: datetime = kwargs.get("created_at", datetime.now())
        self.updated_at: datetime = kwargs.get("updated_at", datetime.now())

    @property
    def id(self) -> str:
        return str(self._id)

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
