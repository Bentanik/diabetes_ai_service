"""
Get Retrieved Context Query Handler - Xử lý truy vấn lấy context từ vector database

File này định nghĩa handler để xử lý GetRetrievedContextQuery, thực hiện việc
lấy context từ vector database.
"""

# from ..get_retrieved_context_query import GetRetrievedContextQuery
# from core.cqrs import QueryHandler, QueryRegistry
# from core.result import Result
# from shared.messages import DocumentResult
# from utils import get_logger
# from rag.retriever import Retriever


# @QueryRegistry.register_handler(GetRetrievedContextQuery)
# class GetRetrievedContextQueryHandler(QueryHandler[Result[RerankResult]]):
#     """
#     Handler xử lý truy vấn GetRetrievedContextQuery để lấy context từ vector database.
#     """

#     def __init__(self):
#         """
#         Khởi tạo handler
#         """
#         super().__init__()
#         self.logger = get_logger(__name__)
#         self.retriever = Retriever()

#     async def execute(self, query: GetRetrievedContextQuery) -> Result[RerankResult]:
#         """
#         Thực thi truy vấn lấy context từ vector database

#         Args:
#             query (GetRetrievedContextQuery): Query chứa query cần lấy context

#         Returns:
#             Result[DocumentModelDTO]: Kết quả thành công hoặc lỗi với message và code tương ứng
#         """
#         try:
#             retrieved_context = await self.retriever.retrieve(
#                 query=query.query,
#                 top_k=10,
#                 rerank_top_n=10,
#                 min_score=0.5,
#             )

#             # Trả về kết quả thành công
#             return Result.success(
#                 message=DocumentResult.FETCHED.message,
#                 code=DocumentResult.FETCHED.code,
#                 data=retrieved_context,
#             )

#         except Exception as e:
#             self.logger.error(f"Lỗi khi lấy tài liệu theo ID: {e}", exc_info=True)
#             return Result.failure(message="Lỗi hệ thống", code="error")


from core.cqrs import QueryHandler, QueryRegistry
from core.result import Result
from ..get_retrieved_context_query import GetRetrievedContextQuery


@QueryRegistry.register_handler(GetRetrievedContextQuery)
class GetRetrievedContextQueryHandler(QueryHandler[Result[None]]):
    pass
