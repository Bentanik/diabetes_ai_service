"""
Get Knowledges Query Handler - Xử lý truy vấn lấy danh sách cơ sở tri thức

File này định nghĩa handler để xử lý GetKnowledgesQuery, thực hiện việc
lấy danh sách cơ sở tri thức từ database với các tùy chọn tìm kiếm,
phân trang và sắp xếp.
"""

from typing import List
from app.database import get_collections
from app.database.models import KnowledgeModel
from app.dto.models import KnowledgeModelDTO
from app.dto.pagination import Pagination
from ..get_knowledges_query import GetKnowledgesQuery
from core.cqrs import QueryRegistry, QueryHandler
from core.result import Result
from shared.messages import KnowledgeMessage
from utils import get_logger


@QueryRegistry.register_handler(GetKnowledgesQuery)
class GetKnowledgesQueryHandler(QueryHandler[Result[Pagination[KnowledgeModelDTO]]]):
    """
    Handler xử lý truy vấn GetKnowledgesQuery để lấy danh sách cơ sở tri thức.
    """

    def __init__(self):
        """
        Khởi tạo handler
        """
        super().__init__()
        self.logger = get_logger(__name__)

    async def execute(
        self, query: GetKnowledgesQuery
    ) -> Result[Pagination[KnowledgeModelDTO]]:
        """
        Thực thi truy vấn lấy danh sách cơ sở tri thức

        Args:
            query (GetKnowledgesQuery): Query chứa các tham số tìm kiếm và phân trang

        Returns:
            Result[Pagination[KnowledgeDTO]]: Kết quả thành công hoặc lỗi với message và code tương ứng
        """
        try:
            self.logger.info(f"Lấy danh sách cơ sở tri thức: {query}")

            # Lấy collection từ database
            collection = get_collections()

            # Xây dựng query tìm kiếm
            filter_query = {}
            if query.search:
                filter_query["name"] = {"$regex": query.search, "$options": "i"}

            if query.select_training is not None:
                filter_query["select_training"] = query.select_training

            # Đếm tổng số bản ghi thỏa mãn điều kiện
            total = await collection.knowledges.count_documents(filter_query)

            # Xây dựng tiêu chí sắp xếp
            sort_direction = -1 if query.sort_order == "desc" else 1
            sort_criteria = [(query.sort_by, sort_direction)]

            # Thực hiện truy vấn với phân trang
            cursor = (
                collection.knowledges.find(filter_query)
                .sort(sort_criteria)
                .skip((query.page - 1) * query.limit)
                .limit(query.limit)
            )

            # Chuyển đổi kết quả sang DTO
            knowledge_dtos: List[KnowledgeModelDTO] = []
            async for doc in cursor:
                knowledge_model = KnowledgeModel.from_dict(doc)
                knowledge_dto = KnowledgeModelDTO.from_model(knowledge_model)
                knowledge_dtos.append(knowledge_dto)

            # Tạo đối tượng phân trang

            total_pages = (total + query.limit - 1) // query.limit
            pagination = Pagination(
                items=knowledge_dtos,
                total=total,
                page=query.page,
                limit=query.limit,
                total_pages=total_pages,
            )

            # Trả về kết quả thành công
            return Result.success(
                message=KnowledgeMessage.FETCHED.message,
                code=KnowledgeMessage.FETCHED.code,
                data=pagination,
            )
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy danh sách: {e}", exc_info=True)
            return Result.failure(message="Lỗi hệ thống", code="error", data=None)
