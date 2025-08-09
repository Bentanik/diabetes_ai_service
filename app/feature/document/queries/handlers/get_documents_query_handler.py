"""
Get Documents Query Handler - Xử lý truy vấn lấy danh sách tài liệu

File này định nghĩa handler để xử lý GetDocumentsQuery, thực hiện việc
lấy danh sách tài liệu với tính năng tìm kiếm, phân trang và sắp xếp.
"""

from typing import Dict, Any, List
from bson import ObjectId
from app.database import get_collections
from app.database.models import DocumentModel
from app.dto.models import DocumentModelDTO
from app.dto.pagination import Pagination
from ..get_documents_query import GetDocumentsQuery
from core.cqrs import QueryHandler, QueryRegistry
from core.result import Result
from shared.messages import DocumentResult
from utils import get_logger


@QueryRegistry.register_handler(GetDocumentsQuery)
class GetDocumentsQueryHandler(
    QueryHandler[Result[Pagination[List[DocumentModelDTO]]]]
):
    """
    Handler xử lý truy vấn GetDocumentsQuery để lấy danh sách tài liệu.
    """

    def __init__(self):
        """
        Khởi tạo handler
        """
        super().__init__()
        self.logger = get_logger(__name__)

    async def execute(
        self, query: GetDocumentsQuery
    ) -> Result[Pagination[List[DocumentModelDTO]]]:
        """
        Thực thi truy vấn lấy danh sách tài liệu

        Args:
            query (GetDocumentsQuery): Query chứa các tham số tìm kiếm và phân trang

        Returns:
            Result[PaginationDTO[List[DocumentModelDTO]]]: Kết quả phân trang với danh sách tài liệu
        """
        try:
            self.logger.info(
                f"Lấy danh sách tài liệu - knowledge_id={query.knowledge_id}, "
                f"search={query.search}, page={query.page}"
            )

            # Xây dựng filter query
            filter_query = self._build_filter_query(query)

            # Xây dựng sort query
            sort_query = self._build_sort_query(query)

            # Tính toán offset cho phân trang
            offset = (query.page - 1) * query.limit

            # Truy vấn database
            collections = get_collections()

            # Lấy tổng số bản ghi thỏa mãn điều kiện
            total_count = await collections.documents.count_documents(filter_query)

            # Lấy dữ liệu với phân trang
            cursor = (
                collections.documents.find(filter_query)
                .sort(sort_query)
                .skip(offset)
                .limit(query.limit)
            )
            docs = await cursor.to_list(length=query.limit)

            # Chuyển đổi sang DTO
            document_dtos = []
            for doc in docs:
                model = DocumentModel.from_dict(doc)
                dto = DocumentModelDTO.from_model(model)
                document_dtos.append(dto)

            # Tính toán số trang
            total_pages = (total_count + query.limit - 1) // query.limit

            # Tạo pagination result
            pagination_result = Pagination(
                items=document_dtos,
                total=total_count,
                page=query.page,
                limit=query.limit,
                total_pages=total_pages,
            )

            return Result.success(
                message=DocumentResult.FETCHED.message,
                code=DocumentResult.FETCHED.code,
                data=pagination_result,
            )

        except Exception as e:
            self.logger.error(f"Lỗi khi lấy danh sách tài liệu: {e}", exc_info=True)
            return Result.failure(message="Lỗi hệ thống", code="error")

    def _build_filter_query(self, query: GetDocumentsQuery) -> Dict[str, Any]:
        """
        Xây dựng filter query cho MongoDB

        Args:
            query (GetDocumentsQuery): Query parameters

        Returns:
            Dict[str, Any]: MongoDB filter query
        """
        filter_query = {}

        # Filter theo knowledge_id nếu có
        if query.knowledge_id and query.knowledge_id.strip():
            if ObjectId.is_valid(query.knowledge_id):
                filter_query["knowledge_id"] = query.knowledge_id
            else:
                self.logger.warning(
                    f"Invalid knowledge_id format: {query.knowledge_id}"
                )

        # Tìm kiếm theo title (case-insensitive)
        if query.search and query.search.strip():
            search_term = query.search.strip()
            filter_query["title"] = {"$regex": search_term, "$options": "i"}

        return filter_query

    def _build_sort_query(self, query: GetDocumentsQuery) -> List[tuple]:
        """
        Xây dựng sort query cho MongoDB

        Args:
            query (GetDocumentsQuery): Query parameters

        Returns:
            List[tuple]: MongoDB sort query
        """
        # Validate sort field để tránh injection
        allowed_sort_fields = [
            "title",
            "created_at",
            "updated_at",
            "priority_diabetes",
            "file_size_bytes",
        ]

        sort_field = (
            query.sort_by if query.sort_by in allowed_sort_fields else "updated_at"
        )
        sort_direction = 1 if query.sort_order.lower() == "asc" else -1

        return [(sort_field, sort_direction)]
