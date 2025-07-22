from .connection import MongoDBConnection
from .db_collections import DBCollections
from .collections_config import initialize_collections_and_indexes
from .manager import (
    connect_to_mongodb,
    close_mongodb_connection,
    get_database,
    get_client,
    check_database_health,
    initialize_database,
    get_collections,
)

__all__ = [
    "MongoDBConnection",
    "connect_to_mongodb",
    "close_mongodb_connection",
    "get_database",
    "get_client",
    "check_database_health",
    "initialize_database",
    "DBCollections",
    "initialize_collections_and_indexes",
    "get_collections",
]
