"""
Setting Model - Module quản lý cài đặt

File này định nghĩa SettingModel để lưu trữ và quản lý thông tin
về các cài đặt trong hệ thống.
"""

from typing import Any, Dict, List, Optional

from app.database.models import BaseModel


class SettingModel(BaseModel):
    """
    Model quản lý cài đặt

    Attributes:
        top_k (int): Số lượng câu trong mỗi passage
        search_accuracy (float): Độ chính xác của tìm kiếm
        temperature (float): Temperature của LLM
        max_tokens (int): Số lượng token tối đa
        list_knowledge_ids (list): Danh sách knowledge id
    """

    def __init__(
        self,
        top_k: int,
        search_accuracy: str,
        temperature: str,
        max_tokens: str,
        list_knowledge_ids: Optional[List[str]] = [],
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.top_k = top_k
        self.search_accuracy = search_accuracy
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.list_knowledge_ids = list_knowledge_ids

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SettingModel":
        """Tạo instance từ MongoDB dictionary"""
        if data is None:
            return None

        data = dict(data)

        top_k = data.pop("top_k", "")
        search_accuracy = data.pop("search_accuracy", "")
        temperature = data.pop("temperature", "")
        max_tokens = data.pop("max_tokens", "")
        list_knowledge_ids = data.pop("list_knowledge_ids", [])

        return cls(
            top_k=top_k,
            search_accuracy=search_accuracy,
            temperature=temperature,
            max_tokens=max_tokens,
            list_knowledge_ids=list_knowledge_ids,
            **data,
        )
