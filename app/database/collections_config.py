"""
Collections Configuration - Cấu hình collections và indexes cho MongoDB

File này định nghĩa logic khởi tạo cho các collections trong MongoDB database,
bao gồm việc tạo collections và các indexes cần thiết để tối ưu hiệu suất
truy vấn dữ liệu.

Chức năng chính:
- Tự động tạo collections nếu chưa tồn tại
- Tạo các indexes theo cấu hình từ index_config.py
- Logging quá trình khởi tạo collections và indexes
"""

from motor.motor_asyncio import AsyncIOMotorDatabase
from utils import get_logger

from app.database.index_config import COLLECTION_INDEX_CONFIG

logger = get_logger(__name__)


async def initialize_collections_and_indexes(db: AsyncIOMotorDatabase) -> None:
    """
    Khởi tạo collections và indexes cho database
    
    Function này sẽ:
    1. Kiểm tra các collections hiện có
    2. Tạo collections mới nếu chưa tồn tại
    3. Tạo các indexes theo cấu hình COLLECTION_INDEX_CONFIG
    
    Args:
        db (AsyncIOMotorDatabase): Database connection object
        
    Returns:
        None
        
    Raises:
        Exception: Có thể raise exception khi tạo collection hoặc index
    """
    # Lấy danh sách tất cả collections hiện có trong database
    existing_collections = await db.list_collection_names()
    
    # Duyệt qua từng collection trong cấu hình
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
