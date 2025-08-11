"""
Setting Model - Module quản lý cài đặt

File này định nghĩa SettingModel để lưu trữ và quản lý thông tin
về các cài đặt trong hệ thống.
"""

from typing import Any, Dict, Optional

from app.database.models import BaseModel


class SettingModel(BaseModel):
    """
    Model quản lý cài đặt

    Attributes:
        number_of_passages (str): Số lượng câu trong mỗi passage
        search_accuracy (str): Độ chính xác của tìm kiếm
        list_collection_name (list): Danh sách collection id
    """

    def __init__(
        self,
        number_of_passages: str,
        search_accuracy: Optional[str] = "",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.number_of_passages = number_of_passages
        self.search_accuracy = search_accuracy

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SettingModel":
        """Tạo instance từ MongoDB dictionary"""
        if data is None:
            return None

        data = dict(data)

        number_of_passages = data.pop("number_of_passages", "")
        search_accuracy = data.pop("search_accuracy", "")

        return cls(
            number_of_passages=number_of_passages,
            search_accuracy=search_accuracy,
            **data,
        )
