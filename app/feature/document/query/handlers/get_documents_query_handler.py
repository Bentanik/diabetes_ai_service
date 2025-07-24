from app.database import get_collections
from app.database.models import DocumentModel
from app.dto import DocumentDTO, Pagination
from app.feature.document import GetDocumentsQuery
from core.cqrs import QueryHandler
from core.cqrs import QueryRegistry
from core.result import Result
from shared.messages import DocumentResult
from utils import get_logger


@QueryRegistry.register_handler(GetDocumentsQuery)
class GetDocumentsQueryHandler(QueryHandler[Result[Pagination[DocumentDTO]]]):
    def __init__(self):
        super().__init__()
        self.logger = get_logger(__name__)

    async def execute(
        self, query: GetDocumentsQuery
    ) -> Result[Pagination[DocumentDTO]]:
        try:
            self.logger.info(f"Lấy danh sách tài liệu: {query}")

            collection = get_collections()
            filter_query = {}

            if query.search:
                filter_query["title"] = {"$regex": query.search, "$options": "i"}

            total = await collection.documents.count_documents(filter_query)

            sort_direction = -1 if query.sort_order == "desc" else 1
            sort_criteria = [(query.sort_by, sort_direction)]

            cursor = (
                collection.documents.find(filter_query)
                .sort(sort_criteria)
                .skip((query.page - 1) * query.limit)
                .limit(query.limit)
            )

            items = []
            async for doc in cursor:
                document_model = DocumentModel.from_dict(doc)
                document_dto = DocumentDTO.from_model(document_model)
                items.append(document_dto)

            pagination = Pagination(
                items=items,
                total=total,
                page=query.page,
                limit=query.limit,
            )

            return Result.success(
                message=DocumentResult.FETCHED.message,
                code=DocumentResult.FETCHED.code,
                data=pagination,
            )
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy danh sách tài liệu: {e}", exc_info=True)
            return Result.failure(message="Lỗi hệ thống", code="error", data=None)
