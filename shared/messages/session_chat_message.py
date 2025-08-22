from shared.messages.base import BaseResultCode


class SessionChatMessage(BaseResultCode):
    SESSION_ALREADY_EXISTS = (
        "CHAT_SESSION_ALREADY_EXISTS",
        "Phiên trò chuyện đã tồn tại",
    )
    SESSION_CREATED = (
        "CHAT_SESSION_CREATED",
        "Phiên trò chuyện đã được tạo thành công",
    )
    SESSION_DELETED_SUCCESS = (
        "CHAT_SESSION_DELETED_SUCCESS",
        "Phiên trò chuyện đã được xóa thành công",
    )
    SESSION_DELETED_FAILED = (
        "CHAT_SESSION_DELETED_FAILED",
        "Phiên trò chuyện không tồn tại",
    )
    SESSION_UPDATED_FAILED = (
        "CHAT_SESSION_UPDATED_FAILED",
        "Phiên trò chuyện không tồn tại",
    )
    SESSION_UPDATED_SUCCESS = (
        "CHAT_SESSION_UPDATED_SUCCESS",
        "Phiên trò chuyện đã được cập nhật thành công",
    )
    SESSION_NOT_FOUND = (
        "CHAT_SESSION_NOT_FOUND",
        "Phiên trò chuyện không tồn tại",
    )
    SESSIONS_FETCHED_SUCCESS = (
        "CHAT_SESSIONS_FETCHED_SUCCESS",
        "Phiên trò chuyện đã được lấy thành công",
    )
