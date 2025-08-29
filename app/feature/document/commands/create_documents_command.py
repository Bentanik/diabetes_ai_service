from dataclasses import dataclass, field
from typing import List
from fastapi import UploadFile
from core.cqrs import Command

@dataclass
class CreateDocumentsCommand(Command):
    """
    Tạo nhiều tài liệu mới
    Attributes:
        files: Danh sách tệp tài liệu
        knowledge_id: ID của cơ sở tri thức
        titles: Danh sách tiêu đề tài liệu
        descriptions: Danh sách mô tả tài liệu
    """

    files: List[UploadFile]
    knowledge_id: str
    titles: List[str] = field(default_factory=list)
    descriptions: List[str] = field(default_factory=list)

    def __post_init__(self):
        if len(self.files) != len(self.titles) or len(self.files) != len(self.descriptions):
            raise ValueError("Số lượng files, titles và descriptions không khớp")

        if not self.knowledge_id:
            raise ValueError("ID của cơ sở tri thức không được để trống")

        # Validate và strip từng item
        self.titles = [title.strip() for title in self.titles if title.strip()]
        self.descriptions = [desc.strip() for desc in self.descriptions if desc.strip()]

        if any(not title for title in self.titles):
            raise ValueError("Tất cả tiêu đề tài liệu không được để trống")

        if any(not desc for desc in self.descriptions):
            raise ValueError("Tất cả mô tả tài liệu không được để trống")
