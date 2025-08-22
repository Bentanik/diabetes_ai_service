from dataclasses import dataclass
from typing import Optional
from core.cqrs import Command


@dataclass
class UpdateSettingCommand(Command):
    """
    Command cập nhật cài đặt

    Attributes:
        top_k (Optional[int]): Số lượng câu trong mỗi passage
        temperature (Optional[float]): Nhiệt độ của LLM
        max_tokens (Optional[int]): Số lượng token tối đa
        search_accuracy (Optional[int]): Độ chính xác của tìm kiếm
        list_knowledge_ids (Optional[list[str]]): Danh sách collection id
    """

    top_k: Optional[int] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    search_accuracy: Optional[float] = None
    list_knowledge_ids: Optional[list[str]] = None
