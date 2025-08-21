from enum import Enum

class DocumentStatus(str, Enum):
    NORMAL = "normal"
    DELETED = "deleted"             # Đã xóa
    DUPLICATE = "duplicate"         # Đã trùng
