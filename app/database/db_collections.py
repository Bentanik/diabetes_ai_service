"""
Database Collections - Quản lý các collection trong MongoDB

File này định nghĩa class DBCollections - một wrapper class để quản lý
tất cả các collections trong MongoDB database. Class này cung cấp
các property để truy cập dễ dàng đến các collections khác nhau.
"""

from motor.motor_asyncio import AsyncIOMotorCollection


class DBCollections:
    """
    Quản lý các collection trong database MongoDB.

    Class này đóng vai trò như một facade pattern để truy cập
    các collections trong database. Mỗi collection được expose
    thông qua một property riêng biệt, đảm bảo type safety
    và dễ dàng sử dụng.

    Attributes:
        db: Database connection object từ Motor
    """

    def __init__(self, db):
        """
        Constructor của DBCollections

        Args:
            db: Database connection object (thường là AsyncIOMotorDatabase)
        """
        self.db = db

    @property
    def knowledges(self) -> AsyncIOMotorCollection:
        """
        Collection lưu trữ thông tin cơ sở tri thức

        Returns:
            AsyncIOMotorCollection: Collection knowledges
        """
        return self.db["knowledges"]

    @property
    def document_jobs(self) -> AsyncIOMotorCollection:
        """
        Collection lưu trữ thông tin jobs xử lý tài liệu

        Returns:
            AsyncIOMotorCollection: Collection document_jobs
        """
        return self.db["document_jobs"]

    @property
    def documents(self) -> AsyncIOMotorCollection:
        """
        Collection lưu trữ thông tin tài liệu

        Returns:
            AsyncIOMotorCollection: Collection documents
        """
        return self.db["documents"]

    @property
    def document_chunks(self) -> AsyncIOMotorCollection:
        """
        Collection lưu trữ thông tin document chunks

        Returns:
            AsyncIOMotorCollection: Collection document_chunks
        """
        return self.db["document_chunks"]

    @property
    def chat_sessions(self) -> AsyncIOMotorCollection:
        """
        Collection lưu trữ thông tin phiên trò chuyện
        """
        return self.db["chat_sessions"]

    @property
    def chat_histories(self) -> AsyncIOMotorCollection:
        """
        Collection lưu trữ thông tin lịch sử cuộc trò chuyện

        Returns:
            AsyncIOMotorCollection: Collection chat_histories
        """
        return self.db["chat_histories"]

    @property
    def settings(self) -> AsyncIOMotorCollection:
        """
        Collection lưu trữ thông tin cài đặt

        Returns:
            AsyncIOMotorCollection: Collection settings
        """
        return self.db["settings"]



   
