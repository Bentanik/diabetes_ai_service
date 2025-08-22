from shared.messages.base import BaseResultCode


class ChatMessage(BaseResultCode):
    CHAT_CREATED = (
        "CHAT_CREATED",
        "Cuộc trò chuyện đã được tạo thành công",
    )
    CHAT_CREATED_FAILED = (
        "CHAT_CREATED_FAILED",
        "Cuộc trò chuyện không được tạo",
    )
    CHAT_NOT_FOUND = (
        "CHAT_NOT_FOUND",
        "Cuộc trò chuyện không tồn tại",
    )
    CHAT_HISTORIES_FETCHED_SUCCESS = (
        "CHAT_HISTORIES_FETCHED_SUCCESS",
        "Lịch sử cuộc trò chuyện đã được lấy thành công",
    )
    CHAT_HISTORIES_FETCHED_FAILED = (
        "CHAT_HISTORIES_FETCHED_FAILED",
        "Lịch sử cuộc trò chuyện không tồn tại",
    )
