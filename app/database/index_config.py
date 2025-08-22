"""
Index Configuration - Cấu hình indexes cho MongoDB collections

File này chứa cấu hình cho tất cả các indexes cần thiết trong database.
"""

COLLECTION_INDEX_CONFIG = {
    "knowledges": [
        {"fields": [("name", 1)], "unique": True, "name": "idx_name"},
    ],
    "document_jobs": [
        {"fields": [("title", 1)], "name": "idx_title"},
        {"fields": [("created_at", -1)], "name": "idx_created_at"},
    ],
    "documents": [
        {"fields": [("knowledge_id", 1)], "name": "idx_knowledge_id"},
        {"fields": [("file_hash", 1)], "unique": True, "name": "file_hash_unique_idx"},
        {
            "fields": [("knowledge_id", 1), ("title", 1)],
            "unique": True,
            "name": "knowledge_title_unique_idx",
        },
        {"fields": [("title", 1)], "name": "idx_title"},
    ],
    "document_chunks": [
        {"fields": [("document_id", 1)], "name": "idx_document_id"},
        {"fields": [("is_active", 1)], "name": "idx_is_active"},
    ],
    "chat_sessions": [
        {"fields": [("user_id", 1)], "name": "idx_user_id"},
        {"fields": [("title", 1)], "name": "idx_title"},
    ],
    "chat_histories": [
        {"fields": [("session_id", 1)], "name": "idx_session_id"},
        {"fields": [("created_at", -1)], "name": "idx_created_at"},
    ],
}
