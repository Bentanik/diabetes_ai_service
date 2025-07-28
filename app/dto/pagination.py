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
    """

    items: List[T] = Field(default_factory=list)
    total: int
    page: int
    limit: int

    @property
    def total_pages(self) -> int:
        """
        Tính toán tổng số trang dựa trên tổng số item và giới hạn mỗi trang

        Returns:
            int: Tổng số trang, được làm tròn lên
        """
        return (self.total + self.limit - 1) // self.limit

    class Config:
        arbitrary_types_allowed = True
