from shared.messages.base import BaseResultCode


class DocumentResult(BaseResultCode):
    CREATING = ("DOCUMENT_CREATING", "Tài liệu đang được tạo")
    CREATED = ("DOCUMENT_CREATED", "Tài liệu đã được tạo thành công")
    TITLE_EXISTS = ("DOCUMENT_TITLE_EXISTS", "Tên tài liệu đã tồn tại")
    FETCHED = ("DOCUMENT_FETCHED", "Tài liệu đã được lấy thành công")
    NOT_FOUND = ("DOCUMENT_NOT_FOUND", "Tài liệu không tồn tại")
    NO_UPDATE = ("DOCUMENT_NO_UPDATE", "Không có thay đổi")
    UPDATED = ("DOCUMENT_UPDATED", "Tài liệu đã được cập nhật thành công")
    DELETED = ("DOCUMENT_DELETED", "Tài liệu đã được xóa thành công")
