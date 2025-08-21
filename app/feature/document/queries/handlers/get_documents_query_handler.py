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
from shared.messages import DocumentMessage
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
        try:
            self.logger.info(
                f"Lấy danh sách tài liệu - knowledge_id={query.knowledge_id}, "
                f"search={query.search}, page={query.page}"
            )

            # Validate knowledge_id
            if not query.knowledge_id or not query.knowledge_id.strip():
                return Result.failure("Thiếu knowledge_id", "knowledge_id_required")

            if not ObjectId.is_valid(query.knowledge_id):
                return Result.failure("knowledge_id không hợp lệ", "invalid_knowledge_id")

            collections = get_collections()
            knowledge_id = ObjectId(query.knowledge_id)

            # Kiểm tra knowledge có tồn tại không
            if not await collections.knowledges.find_one({"_id": knowledge_id}):
                return Result.failure(DocumentMessage.NOT_FOUND.code, DocumentMessage.NOT_FOUND.message)

            # Filter + sort
            filter_query = {"knowledge_id": query.knowledge_id}
            if query.search and query.search.strip():
                filter_query["title"] = {"$regex": query.search.strip(), "$options": "i"}

            sort_field = query.sort_by if query.sort_by in [
                "title", "created_at", "updated_at", "priority_diabetes", "file_size_bytes"
            ] else "updated_at"
            sort_query = [(sort_field, 1 if query.sort_order.lower() == "asc" else -1)]

            # Phân trang
            offset = (query.page - 1) * query.limit
            total_count = await collections.documents.count_documents(filter_query)

            docs = await (
                collections.documents.find(filter_query)
                .sort(sort_query)
                .skip(offset)
                .limit(query.limit)
                .to_list(length=query.limit)
            )

            document_dtos = [
                DocumentModelDTO.from_model(DocumentModel.from_dict(doc)) for doc in docs
            ]

            pagination_result = Pagination(
                items=document_dtos,
                total=total_count,
                page=query.page,
                limit=query.limit,
                total_pages=(total_count + query.limit - 1) // query.limit,
            )

            return Result.success(
                message=DocumentMessage.FETCHED.message,
                code=DocumentMessage.FETCHED.code,
                data=pagination_result,
            )

        except Exception as e:
            self.logger.error(f"Lỗi khi lấy danh sách tài liệu: {e}", exc_info=True)
            return Result.failure("Lỗi hệ thống", "error")

