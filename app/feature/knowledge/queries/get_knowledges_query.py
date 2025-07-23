from dataclasses import dataclass, field
from core.cqrs.base import Query
from typing import Optional


@dataclass
class GetKnowledgesQuery(Query):
    """
    Lấy danh sách cơ sở tri thức với các tham số tìm kiếm, phân trang và sắp xếp.

    Attributes:
        search: Từ khóa tìm kiếm tên cơ sở tri thức (optional)
        page: Trang hiện tại (bắt đầu từ 1)
        limit: Số bản ghi trên mỗi trang
        sort_by: Trường dùng để sắp xếp (ví dụ: 'updated_at', 'name')
        sort_order: Thứ tự sắp xếp, 'asc' hoặc 'desc'
    """

    search: Optional[str] = field(default=None)
    page: int = field(default=1)
    limit: int = field(default=10)
    sort_by: str = field(default="updated_at")
    sort_order: str = field(default="desc")

    def __post_init__(self):
        # Validation cơ bản

        if self.page < 1:
            raise ValueError("Page phải lớn hơn hoặc bằng 1")

        if not (1 <= self.limit <= 100):
            raise ValueError("Limit phải nằm trong khoảng 1 đến 100")

        if self.sort_order not in ("asc", "desc"):
            raise ValueError("Sort order phải là 'asc' hoặc 'desc'")

        if self.search is not None:
            self.search = self.search.strip()
            if self.search == "":
                self.search = None

        self.sort_by = self.sort_by.strip()
