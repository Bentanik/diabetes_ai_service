from bson import ObjectId
from app.database import get_collections
from app.database.models import KnowledgeModel
from app.dto import KnowledgeDTO
from app.feature.knowledge import GetKnowledgeQuery
from core.cqrs import QueryHandler, QueryRegistry
from core.result import Result
from shared.messages import KnowledgeResult
from utils import get_logger


@QueryRegistry.register_handler(GetKnowledgeQuery)
class GetKnowledgeQueryHandler(QueryHandler[Result[KnowledgeDTO]]):
    def __init__(self):
        super().__init__()
        self.logger = get_logger(__name__)

    async def execute(self, query: GetKnowledgeQuery) -> Result[KnowledgeDTO]:
        try:
            self.logger.info(f"Lấy cơ sở tri thức theo ID: {query.id}")

            if not ObjectId.is_valid(query.id):
                return Result.failure(
                    message="ID không hợp lệ", code="invalid_id", data=None
                )

            collection = get_collections()
            doc = await collection.knowledges.find_one({"_id": ObjectId(query.id)})

            if not doc:
                return Result.failure(
                    message="Không tìm thấy dữ liệu", code="not_found", data=None
                )

            model = KnowledgeModel.from_dict(doc)
            dto = KnowledgeDTO.from_model(model)

            return Result.success(
                message=KnowledgeResult.FETCHED.message,
                code=KnowledgeResult.FETCHED.code,
                data=dto,
            )

        except Exception as e:
            self.logger.error(f"Lỗi khi lấy theo ID: {e}", exc_info=True)
            return Result.failure(message="Lỗi hệ thống", code="error", data=None)
