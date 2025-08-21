"""
Get Document Parsers Query Handler - Xử lý truy vấn lấy thông tin phân tích tài liệu

File này định nghĩa handler để xử lý GetDocumentParsersQuery, thực hiện việc
lấy danh sách thông tin phân tích của một tài liệu với tính năng phân trang và sắp xếp.
"""

from typing import Dict, Any, List
from bson import ObjectId
from app.database import get_collections
from app.database.models import DocumentChunkModel
from app.dto.models import DocumentChunkModelDTO
from app.dto.models.document_chunk_dto import DocumentChunkModelDTO
from app.dto.pagination import Pagination
from ..get_document_chunks_query import GetDocumentChunksQuery
from core.cqrs import QueryHandler, QueryRegistry
from core.result import Result
from shared.messages import DocumentMessage
from utils import get_logger


@QueryRegistry.register_handler(GetDocumentChunksQuery)
class GetDocumentChunksQueryHandler(
    QueryHandler[Result[Pagination[List[DocumentChunkModelDTO]]]]
):
    def __init__(self):
        super().__init__()
        self.logger = get_logger(__name__)

    async def execute(
        self, query: GetDocumentChunksQuery
    ) -> Result[Pagination[List[DocumentChunkModelDTO]]]:
        try:
            self.logger.info(
                f"Lấy thông tin phân tích tài liệu - document_id={query.document_id}, "
                f"page={query.page}, limit={query.limit}"
            )

            # Validate document_id
            if not query.document_id or not query.document_id.strip():
                return Result.failure("Thiếu document_id", "document_id_required")

            if not ObjectId.is_valid(query.document_id):
                return Result.failure("document_id không hợp lệ", "invalid_document_id")

            collections = get_collections()
            document_id = ObjectId(query.document_id)

            # Check document tồn tại
            if not await collections.documents.find_one({"_id": document_id}):
                return Result.failure(DocumentMessage.NOT_FOUND.code, DocumentMessage.NOT_FOUND.message)

            # Filter + sort
            filter_query = {"document_id": query.document_id}

            sort_field = query.sort_by if query.sort_by in [
                "page", "block_index", "created_at", "updated_at", "content"
            ] else "updated_at"
            sort_query = [(sort_field, 1 if query.sort_order.lower() == "asc" else -1)]

            # Phân trang
            offset = (query.page - 1) * query.limit
            total_count = await collections.document_chunks.count_documents(filter_query)

            docs = await (
                collections.document_chunks.find(filter_query)
                .sort(sort_query)
                .skip(offset)
                .limit(query.limit)
                .to_list(length=query.limit)
            )

            parser_dtos = [
                DocumentChunkModelDTO.from_model(DocumentChunkModel.from_dict(doc))
                for doc in docs
            ]

            pagination_result = Pagination(
                items=parser_dtos,
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
            self.logger.error(f"Lỗi khi lấy thông tin phân tích tài liệu: {e}", exc_info=True)
            return Result.failure("Lỗi hệ thống", "error")