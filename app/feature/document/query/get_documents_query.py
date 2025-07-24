from dataclasses import dataclass
from typing import Optional
from core.cqrs.base import Query


@dataclass
class GetDocumentsQuery(Query):
    """
    Truy vấn danh sách document jobs với điều kiện lọc, phân trang và sắp xếp.
    """

    search: Optional[str] = None
    page: int = 1
    limit: int = 10
    sort_by: str = "created_at"
    sort_order: str = "desc"

    def __post_init__(self):
        self._validate_pagination()
        self._normalize_sort_order()
        self._normalize_search()
        self._normalize_sort_by()

    def _validate_pagination(self):
        if self.page < 1:
            raise ValueError("Page phải lớn hơn hoặc bằng 1")
        if not (1 <= self.limit <= 100):
            raise ValueError("Limit phải nằm trong khoảng 1 đến 100")

    def _normalize_sort_order(self):
        self.sort_order = self.sort_order.lower()
        if self.sort_order not in ("asc", "desc"):
            raise ValueError("Sort order phải là 'asc' hoặc 'desc'")

    def _normalize_search(self):
        if self.search is not None:
            self.search = self.search.strip() or None

    def _normalize_sort_by(self):
        self.sort_by = self.sort_by.strip()
