from bson import ObjectId
from app.database import get_collections
from app.database.models import DocumentModel
from app.dto import DocumentDTO
from app.feature.document import GetDocumentQuery
from core.cqrs import QueryHandler, QueryRegistry
from core.result import Result
from shared.messages.document_message import DocumentResult
from utils import get_logger


@QueryRegistry.register_handler(GetDocumentQuery)
class GetDocumentQueryHandler(QueryHandler[Result[DocumentDTO]]):
    def __init__(self):
        super().__init__()
        self.logger = get_logger(__name__)

    async def execute(self, query: GetDocumentQuery) -> Result[DocumentDTO]:
        try:
            self.logger.info(f"Lấy tài liệu theo ID: {query.id}")

            if not ObjectId.is_valid(query.id):
                return Result.failure(
                    message="ID không hợp lệ", code="invalid_id", data=None
                )

            collection = get_collections()
            doc = await collection.documents.find_one({"_id": ObjectId(query.id)})

            if not doc:
                return Result.failure(
                    message=DocumentResult.NOT_FOUND.message,
                    code=DocumentResult.NOT_FOUND.code,
                )

            model = DocumentModel.from_dict(doc)
            dto = DocumentDTO.from_model(model)

            return Result.success(
                message=DocumentResult.FETCHED.message,
                code=DocumentResult.FETCHED.code,
                data=dto,
            )

        except Exception as e:
            self.logger.error(f"Lỗi khi lấy theo ID: {e}", exc_info=True)
            return Result.failure(message="Lỗi hệ thống", code="error", data=None)
