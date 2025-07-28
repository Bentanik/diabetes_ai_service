"""
Processing Status - Value Object cho trạng thái xử lý

File này định nghĩa ProcessingStatus để lưu trữ và xử lý thông tin về
trạng thái và tiến độ xử lý của một công việc.
"""

from dataclasses import dataclass
from typing import Dict, Union

from app.database.enums import DocumentJobStatus


@dataclass
class ProcessingStatus:
    """
    Value Object chứa thông tin về trạng thái xử lý

    Attributes:
        status (DocumentJobStatus): Trạng thái hiện tại của công việc
        progress (float): Tiến độ hoàn thành (0.0 - 1.0)
        message (str): Thông báo về tiến độ hoặc lỗi
    """
    status: DocumentJobStatus = DocumentJobStatus.PENDING
    progress: float = 0.0
    message: str = ""

    def to_dict(self) -> Dict[str, Union[str, float]]:
        """Chuyển đổi sang dictionary"""
        return {
            "status": self.status,
            "progress": self.progress,
            "progress_message": self.message,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Union[str, float]]) -> "ProcessingStatus":
        """Tạo instance từ dictionary"""
        status_val = data.get("status", DocumentJobStatus.PENDING)
        if isinstance(status_val, str):
            status_val = DocumentJobStatus(status_val)

        return cls(
            status=status_val,
            progress=data.get("progress", 0.0),
            message=data.get("progress_message", ""),
        ) 