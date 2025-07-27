"""
Index Configuration - Cấu hình indexes cho MongoDB collections

File này chứa cấu hình cho tất cả các indexes cần thiết trong database.
"""

COLLECTION_INDEX_CONFIG = {
    # Collection "knowledges" - lưu trữ thông tin cơ sở tri thức
    "knowledges": [
        {
            "fields": [("name", 1)],
            "unique": True,
            "name": "name_unique_idx"
        },
    ],
    
    # Collection "documents" - lưu trữ thông tin tài liệu
    "documents": [
        {
            "fields": [("knowledge_id", 1)],
            "name": "idx_knowledge_id"
        },
        {
            "fields": [("file_hash", 1)],
            "unique": True,
            "name": "file_hash_unique_idx"
        },
        {
            "fields": [("knowledge_id", 1), ("title", 1)],
            "unique": True,
            "name": "knowledge_title_unique_idx"
        },
    ],
    
    # Collection "document_jobs" - lưu trữ thông tin jobs xử lý tài liệu
    # "document_jobs": [
    #     {
    #         "fields": [("document_id", 1)],
    #         "name": "idx_document_id"
    #     },
    #     {
    #         "fields": [("status", 1)],  # Index trên status để tìm jobs theo trạng thái
    #         "name": "idx_status"
    #     },
    #     {
    #         "fields": [("created_at", -1)],  # Index trên created_at để sắp xếp theo thời gian tạo
    #         "name": "idx_created_at"
    #     },
    # ],
} 