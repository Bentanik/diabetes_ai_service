from dataclasses import dataclass
from typing import List, Optional
from core.cqrs import Command


@dataclass
class UpdateSetting(Command):
    # Command cập nhật cài đặt
    pass

@dataclass
class UpdateTrainingSettingCommand(Command):
    """
    Command cập nhật cài đặt

    Attributes:
        knowledge_ids (Optional[str]): Số lượng câu trong mỗi passage
    """

    knowledge_ids: Optional[List[str]] = None
