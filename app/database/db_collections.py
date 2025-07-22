from motor.motor_asyncio import AsyncIOMotorCollection


class DBCollections:
    """Quản lý các collection trong database MongoDB."""

    def __init__(self, db):
        self.db = db

    @property
    def knowledges(self) -> AsyncIOMotorCollection:
        return self.db["knowledges"]
