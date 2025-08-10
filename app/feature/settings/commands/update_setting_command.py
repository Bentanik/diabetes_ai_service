from dataclasses import dataclass
from typing import Optional
from core.cqrs import Command


@dataclass
class UpdateSettingCommand(Command):
    """
    Command cập nhật cài đặt

    Attributes:
        number_of_passages (Optional[int]): Số lượng câu trong mỗi passage
        search_accuracy (Optional[int]): Độ chính xác của tìm kiếm
        list_knowledge_id (Optional[list[str]]): Danh sách collection id
    """

    number_of_passages: Optional[float] = None
    search_accuracy: Optional[float] = None
    list_knowledge_id: Optional[list[str]] = None
