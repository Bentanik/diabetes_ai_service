"""
Pagination - Module định nghĩa lớp phân trang generic

File này cung cấp lớp Pagination để xử lý phân trang cho bất kỳ loại dữ liệu nào
"""

from pydantic import BaseModel, Field
from typing import Generic, TypeVar, List

T = TypeVar("T")


class Pagination(BaseModel, Generic[T]):
    """
    Lớp phân trang generic, có thể sử dụng với bất kỳ loại dữ liệu nào.

    Attributes:
        items (List[T]): Danh sách các item của trang hiện tại
        total (int): Tổng số item có trong toàn bộ dữ liệu
        page (int): Số trang hiện tại
        limit (int): Số item tối đa trên mỗi trang
        total_pages (int): Tổng số trang
    """

    items: List[T] = Field(default_factory=list)
    total: int
    page: int
    limit: int
    total_pages: int

    class Config:
        arbitrary_types_allowed = True
