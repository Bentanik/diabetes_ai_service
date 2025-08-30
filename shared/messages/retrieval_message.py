from shared.messages.base import BaseResultCode


class RetrievalMessage(BaseResultCode):
    SETTING_NOT_FOUND = ("SETTING_NOT_FOUND", "Cài đặt không tồn tại")
    LIST_KNOWLEDGE_IDS_EMPTY = ("LIST_KNOWLEDGE_IDS_EMPTY", "Danh sách knowledge ids trống")
    TOP_K_INVALID = ("TOP_K_INVALID", "top_k không hợp lệ")
    SEARCH_ACCURACY_INVALID = ("SEARCH_ACCURACY_INVALID", "search_accuracy không hợp lệ")
    EMBEDDING_FAILED = ("EMBEDDING_FAILED", "Lỗi tạo embedding")
    SEARCH_FAILED = ("SEARCH_FAILED", "Lỗi tìm kiếm")
    RETRIEVAL_FAILED = ("RETRIEVAL_FAILED", "Lỗi lấy kết quả tìm kiếm")
    FETCHED = ("RETRIEVAL_FETCHED", "Kết quả tìm kiếm đã được lấy thành công")
    NOT_FOUND = ("RETRIEVAL_NOT_FOUND", "Không tìm thấy kết quả tìm kiếm")
    QUERY_EMPTY = ("QUERY_EMPTY", "Query trống")
