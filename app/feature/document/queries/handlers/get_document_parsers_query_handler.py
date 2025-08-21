"""
Get Document Parsers Query Handler - Xử lý truy vấn lấy thông tin phân tích tài liệu

File này định nghĩa handler để xử lý GetDocumentParsersQuery, thực hiện việc
lấy danh sách thông tin phân tích của một tài liệu với tính năng phân trang và sắp xếp.
"""

from typing import Dict, Any, List
from bson import ObjectId
from app.database import get_collections
from app.database.models import DocumentParserModel
from app.dto.models import DocumentParserModelDTO
from app.dto.pagination import Pagination
from ..get_document_parsers_query import GetDocumentParsersQuery
from core.cqrs import QueryHandler, QueryRegistry
from core.result import Result
from shared.messages import DocumentMessage
from utils import get_logger


@QueryRegistry.register_handler(GetDocumentParsersQuery)
class GetDocumentParsersQueryHandler(
    QueryHandler[Result[Pagination[List[DocumentParserModelDTO]]]]
):
    """
    Handler xử lý truy vấn GetDocumentParsersQuery để lấy thông tin phân tích tài liệu.
    """

    def __init__(self):
        """
        Khởi tạo handler
        """
        super().__init__()
        self.logger = get_logger(__name__)

    async def execute(
        self, query: GetDocumentParsersQuery
    ) -> Result[Pagination[List[DocumentParserModelDTO]]]:
        """
        Thực thi truy vấn lấy thông tin phân tích tài liệu

        Args:
            query (GetDocumentParsersQuery): Query chứa các tham số phân trang và sắp xếp

        Returns:
            Result[PaginationDTO[List[DocumentParserModelDTO]]]: Kết quả phân trang với danh sách thông tin phân tích
        """
        try:
            self.logger.info(
                f"Lấy thông tin phân tích tài liệu - document_id={query.document_id}, "
                f"page={query.page}, limit={query.limit}"
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
            total_count = await collections.document_parsers.count_documents(filter_query)

            # Lấy dữ liệu với phân trang
            cursor = (
                collections.document_parsers.find(filter_query)
                .sort(sort_query)
                .skip(offset)
                .limit(query.limit)
            )
            docs = await cursor.to_list(length=query.limit)

            # Chuyển đổi sang DTO
            parser_dtos = []
            for doc in docs:
                model = DocumentParserModel.from_dict(doc)
                dto = DocumentParserModelDTO.from_model(model)
                parser_dtos.append(dto)

            # Tính toán số trang
            total_pages = (total_count + query.limit - 1) // query.limit

            # Tạo pagination result
            pagination_result = Pagination(
                items=parser_dtos,
                total=total_count,
                page=query.page,
                limit=query.limit,
                total_pages=total_pages,
            )

            return Result.success(
                message=DocumentMessage.FETCHED.message,
                code=DocumentMessage.FETCHED.code,
                data=pagination_result,
            )

        except Exception as e:
            self.logger.error(f"Lỗi khi lấy thông tin phân tích tài liệu: {e}", exc_info=True)
            return Result.failure(message="Lỗi hệ thống", code="error")

    def _build_filter_query(self, query: GetDocumentParsersQuery) -> Dict[str, Any]:
        """
        Xây dựng filter query cho MongoDB

        Args:
            query (GetDocumentParsersQuery): Query parameters

        Returns:
            Dict[str, Any]: MongoDB filter query
        """
        filter_query = {}

        # Filter theo document_id
        if query.document_id and query.document_id.strip():
            if ObjectId.is_valid(query.document_id):
                filter_query["document_id"] = query.document_id
            else:
                self.logger.warning(
                    f"Invalid document_id format: {query.document_id}"
                )

        return filter_query

    def _build_sort_query(self, query: GetDocumentParsersQuery) -> List[tuple]:
        """
        Xây dựng sort query cho MongoDB

        Args:
            query (GetDocumentParsersQuery): Query parameters

        Returns:
            List[tuple]: MongoDB sort query
        """
        # Validate sort field để tránh injection
        allowed_sort_fields = [
            "page",
            "block_index",
            "created_at",
            "updated_at",
            "content",
        ]

        sort_field = (
            query.sort_by if query.sort_by in allowed_sort_fields else "updated_at"
        )
        sort_direction = 1 if query.sort_order.lower() == "asc" else -1

        return [(sort_field, sort_direction)]
