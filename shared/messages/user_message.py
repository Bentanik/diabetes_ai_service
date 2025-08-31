from shared.messages.base import BaseResultCode


class UserMessage(BaseResultCode):
    CREATED = ("USER_CREATED", "Người dùng đã được tạo thành công")
    UPDATED = ("USER_UPDATED", "Người dùng đã được cập nhật thành công")
    NOT_FOUND = ("USER_NOT_FOUND", "Người dùng không tồn tại")
    DUPLICATE = ("USER_DUPLICATE", "Người dùng đã tồn tại")
    HEALTH_RECORD_FAILED = ("HEALTH_RECORD_FAILED", "Dữ liệu sức khỏe không được tạo")
    HEALTH_RECORD_CREATED = ("HEALTH_RECORD_CREATED", "Dữ liệu sức khỏe đã được tạo thành công")