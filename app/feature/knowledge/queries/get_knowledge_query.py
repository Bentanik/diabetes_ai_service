"""
Get Knowledge Query - Query để lấy thông tin của một cơ sở tri thức dựa trên ID.

File này định nghĩa GetKnowledgeQuery cho phép lấy thông tin chi tiết
của một cơ sở tri thức dựa trên ID.
"""

from dataclasses import dataclass
from core.cqrs import Query


@dataclass
class GetKnowledgeQuery(Query):
    """
    Query để lấy thông tin chi tiết của một cơ sở tri thức dựa trên ID.
    Kế thừa từ lớp cơ sở Query trong module CQRS.

    Attributes:
        id (str): ID của cơ sở tri thức cần truy vấn.
                 ID này phải là một chuỗi không rỗng và tồn tại trong hệ thống.
    """

    id: str

    def __post_init__(self):
        """
        Thực hiện validation cơ bản sau khi khởi tạo

        Raises:
            ValueError: Khi ID trống hoặc không hợp lệ
        """
        if not self.id:
            raise ValueError("ID của cơ sở tri thức không được để trống")
