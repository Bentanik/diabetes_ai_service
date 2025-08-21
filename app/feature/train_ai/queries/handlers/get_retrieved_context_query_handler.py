# """
# Get Retrieved Context Query Handler - Xử lý truy vấn lấy context từ vector database

# File này định nghĩa handler để xử lý GetRetrievedContextQuery, thực hiện việc
# lấy context từ vector database.
# """

# from bson import ObjectId
# from app.database.manager import get_collections
# from app.database.models.setting_model import SettingModel
# from app.dto.models.document_model_dto import DocumentModelDTO
# from rag.vector_store.operations import VectorStoreOperations
# from shared.messages.setting_message import SettingResult
# from ..get_retrieved_context_query import GetRetrievedContextQuery
# from core.cqrs import QueryHandler, QueryRegistry
# from core.result import Result
# from shared.messages import DocumentResult
# from utils import get_logger
# from app.database.models.document_model import DocumentModel
# from typing import List


# @QueryRegistry.register_handler(GetRetrievedContextQuery)
# class GetRetrievedContextQueryHandler(QueryHandler[Result[List[DocumentModelDTO]]]):
#     """
#     Handler xử lý truy vấn GetRetrievedContextQuery để lấy context từ vector database.
#     """

#     def __init__(self):
#         """
#         Khởi tạo handler
#         """
#         super().__init__()
#         self.logger = get_logger(__name__)
#         self.vector_db = VectorStoreOperations.get_instance()
#         self.collections = get_collections()

#     async def execute(
#         self, query: GetRetrievedContextQuery
#     ) -> Result[List[DocumentModelDTO]]:
#         try:
#             # Lấy cài đặt
#             setting = await self.collections.settings.find_one({})
#             if not setting:
#                 return Result.failure(
#                     message=SettingResult.NOT_FOUND.message,
#                     code=SettingResult.NOT_FOUND.code,
#                     data=[],
#                 )

#             setting = SettingModel.from_dict(setting)

#             # Lấy context từ vector database
#             retrieved_context = await self.vector_db.search(
#                 query_text=query.query,
#                 top_k=setting.number_of_passages,
#                 score_threshold=setting.search_accuracy / 100,
#             )

#             # Sử dụng set để loại bỏ duplicate document_ids
#             raw_document_ids = {
#                 context.payload.get("document_id")
#                 for context in retrieved_context
#                 if context.payload.get("document_id")
#             }

#             if not raw_document_ids:
#                 return Result.success(
#                     message=DocumentResult.FETCHED.message,
#                     code=DocumentResult.FETCHED.code,
#                     data=[],
#                 )

#             # Convert sang ObjectId nếu cần
#             document_ids = []
#             for doc_id in raw_document_ids:
#                 try:
#                     document_ids.append(ObjectId(doc_id))
#                 except Exception as e:
#                     self.logger.warning(f"Invalid ObjectId: {doc_id}, error: {e}")
#                     continue

#             if not document_ids:
#                 return Result.success(
#                     message=DocumentResult.FETCHED.message,
#                     code=DocumentResult.FETCHED.code,
#                     data=[],
#                 )

#             # Batch query - lấy tất cả documents trong 1 lần
#             documents_data = await self.collections.documents.find(
#                 {"_id": {"$in": document_ids}}
#             ).to_list(length=len(document_ids))

#             # Convert sang DTO
#             documents = []
#             for doc_data in documents_data:
#                 try:
#                     document_model = DocumentModel.from_dict(doc_data)
#                     document_dto = DocumentModelDTO.from_model(document_model)
#                     documents.append(document_dto)
#                 except Exception as e:
#                     self.logger.warning(
#                         f"Lỗi convert document {doc_data.get('_id')}: {e}"
#                     )
#                     continue

#             # Trả về kết quả thành công
#             return Result.success(
#                 message=DocumentResult.FETCHED.message,
#                 code=DocumentResult.FETCHED.code,
#                 data=documents,
#             )

#         except Exception as e:
#             self.logger.error(f"Lỗi khi lấy tài liệu theo ID: {e}", exc_info=True)
#             return Result.failure(message="Lỗi hệ thống", code="error")

#         except Exception as e:
#             self.logger.error(f"Lỗi khi lấy tài liệu theo ID: {e}", exc_info=True)
#             return Result.failure(message="Lỗi hệ thống", code="error")
