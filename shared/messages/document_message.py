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
    DUPLICATE = ("DOCUMENT_DUPLICATE", "Tài liệu đã tồn tại")
    NO_CONTENT = ("DOCUMENT_NO_CONTENT", "Không có nội dung để xử lý")
    FAILED_TO_PARSE = ("DOCUMENT_FAILED_TO_PARSE", "Lỗi khi xử lý tài liệu")
    FAILED_TO_SAVE = ("DOCUMENT_FAILED_TO_SAVE", "Lỗi khi lưu tài liệu")
    FAILED_TO_DELETE = ("DOCUMENT_FAILED_TO_DELETE", "Lỗi khi xóa tài liệu")
    FAILED_TO_UPDATE = ("DOCUMENT_FAILED_TO_UPDATE", "Lỗi khi cập nhật tài liệu")
    FAILED_TO_FETCH = ("DOCUMENT_FAILED_TO_FETCH", "Lỗi khi lấy tài liệu")
