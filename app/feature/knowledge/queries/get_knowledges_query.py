"""
Get Knowledges Query - Query để lấy danh sách cơ sở tri thức

File này định nghĩa GetKnowledgesQuery cho phép lấy danh sách cơ sở tri thức
với các tùy chọn tìm kiếm, phân trang và sắp xếp.
"""

from dataclasses import dataclass, field
from core.cqrs import Query
from typing import Optional


@dataclass
class GetKnowledgesQuery(Query):
    """
    Query để lấy danh sách cơ sở tri thức với các tham số tìm kiếm, phân trang và sắp xếp.

    Attributes:
        search (Optional[str]): Từ khóa tìm kiếm tên cơ sở tri thức, có thể bỏ trống
        page (int): Số trang hiện tại, bắt đầu từ 1
        limit (int): Số bản ghi tối đa trên mỗi trang, từ 1-100
        sort_by (str): Trường dùng để sắp xếp (mặc định: 'updated_at')
        sort_order (str): Thứ tự sắp xếp ('asc' hoặc 'desc')
    """

    search: Optional[str] = field(default=None)
    page: int = field(default=1)
    limit: int = field(default=10)
    sort_by: str = field(default="updated_at")
    sort_order: str = field(default="desc")
    select_training: Optional[bool] = field(default=None)

    def __post_init__(self):
        """
        Thực hiện validation cơ bản sau khi khởi tạo

        Raises:
            ValueError: Khi các tham số không hợp lệ
        """
        # Validation cơ bản
        if self.page < 1:
            raise ValueError("Page phải lớn hơn hoặc bằng 1")

        if not (1 <= self.limit <= 100):
            raise ValueError("Limit phải nằm trong khoảng 1 đến 100")

        if self.sort_order not in ("asc", "desc"):
            raise ValueError("Sort order phải là 'asc' hoặc 'desc'")

        # Làm sạch dữ liệu tìm kiếm
        if self.search is not None:
            self.search = self.search.strip()
            if self.search == "":
                self.search = None

        # Làm sạch trường sắp xếp
        self.sort_by = self.sort_by.strip()
