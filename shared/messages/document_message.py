from shared.messages.base import BaseResultCode


class DocumentMessage(BaseResultCode):
    CREATING = ("DOCUMENT_CREATING", "Tài liệu đang được tạo")
    CREATED = ("DOCUMENT_CREATED", "Tài liệu đã được tạo thành công")
    TITLE_EXISTS = ("DOCUMENT_TITLE_EXISTS", "Tên tài liệu đã tồn tại")
    FETCHED = ("DOCUMENT_FETCHED", "Tài liệu đã được lấy thành công")
    NOT_FOUND = ("DOCUMENT_NOT_FOUND", "Tài liệu không tồn tại")
    NO_UPDATE = ("DOCUMENT_NO_UPDATE", "Không có thay đổi")
    UPDATED = ("DOCUMENT_UPDATED", "Tài liệu đã được cập nhật thành công")
    DELETED = ("DOCUMENT_DELETED", "Tài liệu đã được xóa thành công")
    DUPLICATE = ("DOCUMENT_DUPLICATE", "Tài liệu đã tồn tại")
    TRAINING_FAILED = ("DOCUMENT_TRAINING_FAILED", "Lỗi khi huấn luyện tài liệu")
    TRAINING_ALREADY_EXISTS = (
        "DOCUMENT_TRAINING_ALREADY_EXISTS",
        "Tài liệu đã được huấn luyện",
    )
    TRAINING_COMPLETED = (
        "DOCUMENT_TRAINING_COMPLETED",
        "Tài liệu đã được huấn luyện thành công",
    )
    TRAINING_STARTED = ("DOCUMENT_TRAINING_STARTED", "Tài liệu bắt đầu huấn luyện")
