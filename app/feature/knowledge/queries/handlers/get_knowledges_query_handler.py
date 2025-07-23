from app.database import get_collections
from app.database.models import KnowledgeModel
from app.dto import KnowledgeDTO, Pagination
from app.feature.knowledge import GetKnowledgesQuery
from core.cqrs import QueryHandler
from core.cqrs import QueryRegistry
from core.result import Result
from shared.messages import KnowledgeResult
from utils import get_logger


@QueryRegistry.register_handler(GetKnowledgesQuery)
class GetKnowledgesQueryHandler(QueryHandler[Result[Pagination[KnowledgeDTO]]]):
    def __init__(self):
        super().__init__()
        self.logger = get_logger(__name__)

    async def execute(
        self, query: GetKnowledgesQuery
    ) -> Result[Pagination[KnowledgeDTO]]:
        try:
            self.logger.info(f"Lấy danh sách cơ sở tri thức: {query}")

            collection = get_collections()

            filter_query = {}
            if query.search:
                filter_query["name"] = {"$regex": query.search, "$options": "i"}

            total = await collection.knowledges.count_documents(filter_query)

            sort_direction = -1 if query.sort_order == "desc" else 1
            sort_criteria = [(query.sort_by, sort_direction)]

            cursor = (
                collection.knowledges.find(filter_query)
                .sort(sort_criteria)
                .skip((query.page - 1) * query.limit)
                .limit(query.limit)
            )

            items = []
            async for doc in cursor:
                knowledge_model = KnowledgeModel.from_dict(doc)
                knowledge_dto = KnowledgeDTO.from_model(knowledge_model)
                items.append(knowledge_dto)

            pagination = Pagination(
                items=items,
                total=total,
                page=query.page,
                limit=query.limit,
            )

            return Result.success(
                message=KnowledgeResult.FETCHED.message,
                code=KnowledgeResult.FETCHED.code,
                data=pagination,
            )
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy danh sách: {e}", exc_info=True)
            return Result.failure(message="Lỗi hệ thống", code="error", data=None)
