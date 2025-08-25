from typing import List
from bson import ObjectId
from app.database import get_collections
from app.database.models import DocumentChunkModel
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
                return Result.failure(
                    DocumentMessage.NOT_FOUND.message,
                    DocumentMessage.NOT_FOUND.code,
                )

            # Filter
            filter_query = {"document_id": query.document_id}

            if query.min_diabetes_score is not None:
                filter_query["diabetes_score"] = {"$gte": query.min_diabetes_score}
            if query.max_diabetes_score is not None:
                filter_query.setdefault("diabetes_score", {})
                filter_query["diabetes_score"]["$lte"] = query.max_diabetes_score

            # Sort với tie-breaker là _id để đảm bảo ổn định
            sort_field = (
                query.sort_by
                if query.sort_by in ["page", "block_index", "created_at", "updated_at", "content"]
                else "updated_at"
            )
            sort_order = (query.sort_order or "desc").lower()
            sort_direction = 1 if sort_order == "asc" else -1

            # ✅ Thêm _id để đảm bảo không bị trùng/trượt trang
            sort_query = [
                (sort_field, sort_direction),
                ("_id", 1)  # Tie-breaker: đảm bảo thứ tự nhất quán
            ]

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

            total_pages = (total_count + query.limit - 1) // query.limit

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
            return Result.failure("Lỗi hệ thống", "error")