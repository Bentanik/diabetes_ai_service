from typing import List
from app.database import get_collections
from core.cqrs import QueryHandler, QueryRegistry
from core.embedding import EmbeddingModel
from core.result import Result
from shared.messages import RetrievalMessage
from ..get_retrieved_context_query import GetRetrievedContextQuery
from app.dto.models import SearchDocumentDTO
from app.database.models import DocumentModel, SettingModel
from rag.retrieval.retriever import Retriever
from bson import ObjectId
from utils import get_logger


@QueryRegistry.register_handler(GetRetrievedContextQuery)
class GetRetrievedContextQueryHandler(QueryHandler[Result[List[SearchDocumentDTO]]]):

    def __init__(self):
        self.embedding_service = EmbeddingModel()
        self.db = get_collections()
        self.logger = get_logger(__name__)
        self.retriever = None

    async def execute(self, query: GetRetrievedContextQuery) -> Result[List[SearchDocumentDTO]]:
        try:
            # Bước 1: Lấy setting
            setting_doc = await self.db.settings.find_one({})
            if not setting_doc:
                return Result.failure(
                    code=RetrievalMessage.SETTING_NOT_FOUND.code,
                    message=RetrievalMessage.SETTING_NOT_FOUND.message
                )

            setting = SettingModel.from_dict(setting_doc)

            # Validate setting
            if not setting.list_knowledge_ids:
                return Result.failure(
                    code=RetrievalMessage.LIST_KNOWLEDGE_IDS_EMPTY.code,
                    message=RetrievalMessage.LIST_KNOWLEDGE_IDS_EMPTY.message
                )
            if setting.top_k <= 0:
                return Result.failure(
                    code=RetrievalMessage.TOP_K_INVALID.code,
                    message=RetrievalMessage.TOP_K_INVALID.message
                )
            if not (0.0 <= setting.search_accuracy <= 1.0):
                return Result.failure(
                    code=RetrievalMessage.SEARCH_ACCURACY_INVALID.code,
                    message=RetrievalMessage.SEARCH_ACCURACY_INVALID.message
                )

            # Bước 2: Tạo embedding
            query_vector = await self.embedding_service.embed(query.query)
            if not query_vector:
                return Result.failure(
                    code=RetrievalMessage.EMBEDDING_FAILED.code,
                    message=RetrievalMessage.EMBEDDING_FAILED.message
                )

            # Bước 3: Dùng Retriever để tìm kiếm
            if self.retriever is None or self.retriever.collections != setting.list_knowledge_ids:
                self.retriever = Retriever(collections=setting.list_knowledge_ids)

            search_results = await self.retriever.retrieve(
                query_vector=query_vector,
                top_k=setting.top_k,
                score_threshold=setting.search_accuracy,
                document_is_active=True,
                metadata__is_active=True
            )
            
            if not search_results:
                return Result.success(code=RetrievalMessage.NOT_FOUND.code, message=RetrievalMessage.NOT_FOUND.message)

            # Bước 4: Lấy thông tin từ kết quả
            doc_ids = [hit["payload"].get("metadata", {}).get("document_id") for hit in search_results]
            contents = [hit["payload"].get("content") for hit in search_results]
            scores = [hit["score"] for hit in search_results]
            unique_doc_ids = list(dict.fromkeys(doc_ids))
            
            # Bước 5: Lấy thông tin từ DB
            cursor = self.db.documents.find({
                "_id": {"$in": [ObjectId(doc_id) for doc_id in unique_doc_ids if ObjectId.is_valid(doc_id)]}
            })
            document_docs = await cursor.to_list(length=None)
            documents = [DocumentModel.from_dict(doc) for doc in document_docs]
            
            # Bước 6: Tạo DTO
            result_dtos: List[SearchDocumentDTO] = []
            for index, doc_id in enumerate(doc_ids):
                index_doc = unique_doc_ids.index(doc_id)
                dto = SearchDocumentDTO.from_model(
                    model=documents[index_doc],
                    content=contents[index],
                    score=scores[index]
                )
                result_dtos.append(dto)

            return Result.success(code=RetrievalMessage.FETCHED.code, message=RetrievalMessage.FETCHED.message, data=result_dtos)

        except Exception as e:
            self.logger.error(f"Error in GetRetrievedContextQueryHandler: {e}", exc_info=True)
            return Result.failure(
                code=RetrievalMessage.RETRIEVAL_FAILED.code,
                message=RetrievalMessage.RETRIEVAL_FAILED.message
            )