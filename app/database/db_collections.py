from motor.motor_asyncio import AsyncIOMotorCollection


class DBCollections:
    """Quản lý các collection trong database MongoDB."""

    def __init__(self, db):
        self.db = db

    @property
    def knowledges(self) -> AsyncIOMotorCollection:
        return self.db["knowledges"]

    @property
    def documents(self) -> AsyncIOMotorCollection:
        return self.db["documents"]

    @property
    def documents_parsers(self) -> AsyncIOMotorCollection:
        return self.db["documents_parsers"]

    @property
    def document_jobs(self) -> AsyncIOMotorCollection:
        return self.db["document_jobs"]
