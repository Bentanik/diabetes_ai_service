"""
Processing Status - Value Object cho trạng thái xử lý

File này định nghĩa ProcessingStatus để lưu trữ và xử lý thông tin về
trạng thái và tiến độ xử lý của một công việc.
"""

from dataclasses import dataclass
from typing import Dict, Any

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
    progress_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi sang dictionary cho MongoDB"""
        return {
            "status": self.status,
            "progress": self.progress,
            "progress_message": self.progress_message,
        }
