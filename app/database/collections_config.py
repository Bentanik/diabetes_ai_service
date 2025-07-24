from motor.motor_asyncio import AsyncIOMotorDatabase
from utils import get_logger

logger = get_logger(__name__)

COLLECTION_INDEX_CONFIG = {
    "knowledges": [
        {"fields": [("name", 1)], "unique": True, "name": "name_unique_idx"},
    ],
    "documents": [
        {"fields": [("knowledge_id", 1)], "name": "idx_knowledge_id"},
        {"fields": [("file_hash", 1)], "name": "idx_file_hash"},
        {
            "fields": [("knowledge_id", 1), ("title", 1)],
            "unique": True,
            "name": "uniq_knowledge_title",
        },
    ],
}


async def initialize_collections_and_indexes(db: AsyncIOMotorDatabase) -> None:
    existing_collections = await db.list_collection_names()
    for col_name, indexes in COLLECTION_INDEX_CONFIG.items():
        # Tạo collection nếu chưa có
        if col_name not in existing_collections:
            try:
                await db.create_collection(col_name)
                logger.info(f"Đã tạo collection: {col_name}")
            except Exception as e:
                logger.warning(f"Collection {col_name} đã tồn tại hoặc lỗi khác: {e}")

        collection = db[col_name]
        # Tạo các index theo config
        for idx in indexes:
            try:
                await collection.create_index(
                    idx["fields"],
                    unique=idx.get("unique", False),
                    name=idx.get("name"),
                )
                logger.info(
                    f"Đã tạo index cho {col_name}: {idx['fields']} (unique={idx.get('unique', False)})"
                )
            except Exception as e:
                logger.error(f"Lỗi tạo index {idx['fields']} cho {col_name}: {e}")
