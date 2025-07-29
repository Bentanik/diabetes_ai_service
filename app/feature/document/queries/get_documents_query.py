"""
Get Documents Query - Truy vấn lấy danh sách tài liệu

File này định nghĩa GetDocumentsQuery để lấy danh sách tài liệu
với tính năng tìm kiếm, phân trang và sắp xếp.
"""

from dataclasses import dataclass
from typing import Optional
from core.cqrs import Query


@dataclass
class GetDocumentsQuery(Query):
    """
    Query lấy danh sách tài liệu với phân trang và tìm kiếm

    Attributes:
        knowledge_id (Optional[str]): ID của knowledge để filter (nếu có)
        search (str): Từ khóa tìm kiếm theo title
        page (int): Số trang hiện tại (bắt đầu từ 1)
        limit (int): Số lượng bản ghi mỗi trang
        sort_by (str): Trường dùng để sắp xếp
        sort_order (str): Thứ tự sắp xếp (asc/desc)
    """

    knowledge_id: Optional[str] = None
    search: str = ""
    page: int = 1
    limit: int = 10
    sort_by: str = "updated_at"
    sort_order: str = "desc"
