"""
Get Document Query Handler - Xử lý truy vấn lấy thông tin tài liệu

File này định nghĩa handler để xử lý GetDocumentQuery, thực hiện việc
lấy thông tin chi tiết của một tài liệu từ database dựa trên ID.
"""

from bson import ObjectId
from app.database import get_collections
from app.database.models import DocumentModel
from app.dto.models import DocumentModelDTO
from ..get_document_query import GetDocumentQuery
from core.cqrs import QueryHandler, QueryRegistry
from core.result import Result
from shared.messages import DocumentMessage
from utils import get_logger


@QueryRegistry.register_handler(GetDocumentQuery)
class GetDocumentQueryHandler(QueryHandler[Result[DocumentModelDTO]]):
    """
    Handler xử lý truy vấn GetDocumentQuery để lấy thông tin tài liệu.
    """

    def __init__(self):
        """
        Khởi tạo handler
        """
        super().__init__()
        self.logger = get_logger(__name__)

    async def execute(self, query: GetDocumentQuery) -> Result[DocumentModelDTO]:
        """
        Thực thi truy vấn lấy thông tin tài liệu

        Args:
            query (GetDocumentQuery): Query chứa ID của tài liệu cần lấy

        Returns:
            Result[DocumentModelDTO]: Kết quả thành công hoặc lỗi với message và code tương ứng
        """
        try:
            self.logger.info(f"Lấy tài liệu theo ID: {query.id}")

            # Kiểm tra tính hợp lệ của ID
            if not ObjectId.is_valid(query.id):
                return Result.failure(message="ID không hợp lệ", code="invalid_id")

            # Truy vấn database
            collection = get_collections()
            doc = await collection.documents.find_one({"_id": ObjectId(query.id)})

            # Kiểm tra kết quả tìm kiếm
            if not doc:
                return Result.failure(
                    message=DocumentMessage.NOT_FOUND.message,
                    code=DocumentMessage.NOT_FOUND.code,
                )

            # Chuyển đổi dữ liệu sang DTO
            model = DocumentModel.from_dict(doc)
            dto = DocumentModelDTO.from_model(model)

            # Trả về kết quả thành công
            return Result.success(
                message=DocumentMessage.FETCHED.message,
                code=DocumentMessage.FETCHED.code,
                data=dto,
            )

        except Exception as e:
            self.logger.error(f"Lỗi khi lấy tài liệu theo ID: {e}", exc_info=True)
            return Result.failure(message="Lỗi hệ thống", code="error")
