"""
Get Document Parser Query - Truy vấn lấy thông tin phân tích tài liệu

File này định nghĩa GetDocumentParserQuery để lấy thông tin phân tích
của một tài liệu từ database dựa trên ID.
"""

from dataclasses import dataclass
from core.cqrs import Query


@dataclass
class GetDocumentChunksQuery(Query):
    """
    Query lấy thông tin phân tích tài liệu

    Attributes:
        document_id (str): ID của tài liệu cần lấy thông tin
        min_diabetes_score (float): Điểm diabetes tối thiểu
        max_diabetes_score (float): Điểm diabetes tối đa
        page (int): Số trang hiện tại (bắt đầu từ 1)
        limit (int): Số lượng bản ghi mỗi trang
        sort_by (str): Trường dùng để sắp xếp
        sort_order (str): Thứ tự sắp xếp (asc/desc)
    """

    document_id: str
    min_diabetes_score: float = None
    max_diabetes_score: float = None
    page: int = 1
    limit: int = 10
    sort_by: str = "updated_at"
    sort_order: str = "desc"
