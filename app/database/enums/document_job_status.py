from enum import Enum


class DocumentJobStatus(str, Enum):
    """
    Enum định nghĩa các trạng thái của công việc

    Values:
        PENDING: Đang chờ xử lý
        PROCESSING: Đang trong quá trình xử lý
        COMPLETED: Đã xử lý hoàn thành
        FAILED: Xử lý thất bại
    """

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
