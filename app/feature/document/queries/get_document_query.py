"""
Get Document Query - Truy vấn lấy thông tin tài liệu

File này định nghĩa GetDocumentQuery để lấy thông tin chi tiết
của một tài liệu từ database dựa trên ID.
"""

from dataclasses import dataclass
from core.cqrs import Query


@dataclass
class GetDocumentQuery(Query):
    """
    Query lấy thông tin chi tiết một tài liệu

    Attributes:
        id (str): ID của tài liệu cần lấy thông tin
    """

    id: str
