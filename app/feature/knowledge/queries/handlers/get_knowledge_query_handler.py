"""
Get Knowledge Query Handler - Xử lý truy vấn lấy thông tin cơ sở tri thức

File này định nghĩa handler để xử lý GetKnowledgeQuery, thực hiện việc
lấy thông tin chi tiết của một cơ sở tri thức từ database dựa trên ID.
"""

from bson import ObjectId
from app.database import get_collections
from app.database.models import KnowledgeModel
from app.dto.models import KnowledgeModelDTO
from ..get_knowledge_query import GetKnowledgeQuery
from core.cqrs import QueryHandler, QueryRegistry
from core.result import Result
from shared.messages import KnowledgeMessage
from utils import get_logger


@QueryRegistry.register_handler(GetKnowledgeQuery)
class GetKnowledgeQueryHandler(QueryHandler[Result[KnowledgeModelDTO]]):
    """
    Handler xử lý truy vấn GetKnowledgeQuery để lấy thông tin cơ sở tri thức.
    """

    def __init__(self):
        """
        Khởi tạo handler
        """
        super().__init__()
        self.logger = get_logger(__name__)

    async def execute(self, query: GetKnowledgeQuery) -> Result[KnowledgeModelDTO]:
        """
        Thực thi truy vấn lấy thông tin cơ sở tri thức

        Args:
            query (GetKnowledgeQuery): Query chứa ID của cơ sở tri thức cần lấy

        Returns:
            Result[KnowledgeDTO]: Kết quả thành công hoặc lỗi với message và code tương ứng
        """
        try:
            self.logger.info(f"Lấy cơ sở tri thức theo ID: {query.id}")

            # Kiểm tra tính hợp lệ của ID
            if not ObjectId.is_valid(query.id):
                return Result.failure(
                    message="ID không hợp lệ", code="invalid_id"
                )

            # Truy vấn database
            collection = get_collections()
            doc = await collection.knowledges.find_one({"_id": ObjectId(query.id)})

            # Kiểm tra kết quả tìm kiếm
            if not doc:
                return Result.failure(
                    message=KnowledgeMessage.NOT_FOUND.message,
                    code=KnowledgeMessage.NOT_FOUND.code,
                )

            # Chuyển đổi dữ liệu sang DTO
            model = KnowledgeModel.from_dict(doc)
            dto = KnowledgeModelDTO.from_model(model)

            # Trả về kết quả thành công
            return Result.success(
                message=KnowledgeMessage.FETCHED.message,
                code=KnowledgeMessage.FETCHED.code,
                data=dto,
            )

        except Exception as e:
            self.logger.error(f"Lỗi khi lấy theo ID: {e}", exc_info=True)
            return Result.failure(message="Lỗi hệ thống", code="error", data=None)
