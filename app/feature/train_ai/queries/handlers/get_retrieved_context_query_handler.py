from typing import List
from app.database import get_collections
from core.cqrs import QueryHandler, QueryRegistry
from core.embedding.embedding_model import EmbeddingModel
from core.embedding import RerankModel
from core.llm import QwenLLM
from core.result import Result
from rag.retrieval.retriever import Retriever
from shared.messages import RetrievalMessage
from ..get_retrieved_context_query import GetRetrievedContextQuery
from app.dto.models import SearchDocumentDTO
from app.database.models import DocumentModel, SettingModel
from bson import ObjectId
from utils import get_logger
import statistics

@QueryRegistry.register_handler(GetRetrievedContextQuery)
class GetRetrievedContextQueryHandler(QueryHandler[Result[List[SearchDocumentDTO]]]):
    def __init__(self):
        self.embedding_model = None
        self.rerank_model = None
        self.llm = None
        self.db = get_collections()
        self.logger = get_logger(__name__)

    async def _get_embedding_model(self):
        if self.embedding_model is None:
            self.embedding_model = await EmbeddingModel.get_instance()
        return self.embedding_model

    async def _get_rerank_model(self):
        if self.rerank_model is None:
            self.rerank_model = await RerankModel.get_instance()
        return self.rerank_model

    async def _get_llm(self):
        if self.llm is None:
            self.llm = QwenLLM()
        return self.llm

    async def execute(self, query: GetRetrievedContextQuery) -> Result[List[SearchDocumentDTO]]:
        try:
            self.logger.info(f"Bắt đầu xử lý query: '{query.query}'")

            # 1. Lấy cài đặt
            setting_doc = await self.db.settings.find_one({})
            if not setting_doc:
                return Result.failure(
                    code=RetrievalMessage.SETTING_NOT_FOUND.code,
                    message=RetrievalMessage.SETTING_NOT_FOUND.message
                )

            setting = SettingModel.from_dict(setting_doc)

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

            # 2. Lấy model
            embedding_model = await self._get_embedding_model()
            rerank_model = await self._get_rerank_model()
            llm = await self._get_llm()

            # 3. Tiền xử lý query
            query_text = query.query.strip()
            if not query_text:
                return Result.failure(
                    code=RetrievalMessage.QUERY_EMPTY.code,
                    message=RetrievalMessage.QUERY_EMPTY.message
                )

            # 4. Tạo embedding
            query_vector = await embedding_model.embed(query_text.lower())
            if not query_vector:
                return Result.failure(
                    code=RetrievalMessage.EMBEDDING_FAILED.code,
                    message=RetrievalMessage.EMBEDDING_FAILED.message
                )

            # 5. Lấy external synonyms từ setting
            external_synonyms = getattr(setting, 'synonyms', {}) or {
                "tiểu đường": ["đái tháo đường", "diabetes", "bệnh tiểu đường"],
                "đái tháo đường": ["tiểu đường", "diabetes", "bệnh đái tháo đường"],
                "đái tháo đường loại 2": ["tiểu đường loại 2", "diabetes type 2", "bệnh đái tháo đường loại 2"]
            }
            self.logger.info(f"External synonyms: {external_synonyms}")

            # 6. Tạo Retriever
            retriever = Retriever(
                collections=setting.list_knowledge_ids,
                embedding_model=embedding_model,
                rerank_model=rerank_model,
                llm=llm,
                external_synonyms=external_synonyms,
                use_rerank=True,
                initial_top_k=100,
                top_k=setting.top_k
            )

            # 7. Tìm kiếm
            search_results = await retriever.retrieve(
                query_vector=query_vector,
                query_text=query_text,
                score_threshold=setting.search_accuracy,
                document_is_active=True,
                metadata__is_active=True
            )

            self.logger.info(f"Retriever trả về {len(search_results)} kết quả")

            if not search_results:
                return Result.success(
                    data=[],
                    code=RetrievalMessage.NOT_FOUND.code,
                    message=RetrievalMessage.NOT_FOUND.message
                )

            # 8. Lấy thông tin document
            object_ids = []
            for hit in search_results:
                doc_id = hit["payload"].get("metadata", {}).get("document_id")
                if doc_id and ObjectId.is_valid(doc_id):
                    object_ids.append(ObjectId(doc_id))

            if not object_ids:
                return Result.success(data=[], code=RetrievalMessage.NOT_FOUND.code, message="Không tìm thấy document_id hợp lệ")

            cursor = self.db.documents.find({"_id": {"$in": object_ids}})
            docs = await cursor.to_list(length=None)
            document_map = {str(doc["_id"]): DocumentModel.from_dict(doc) for doc in docs}

            # 9. Tạo DTO
            result_dtos = []
            for hit in search_results:
                payload = hit["payload"]
                doc_id = payload.get("metadata", {}).get("document_id")
                content = payload.get("content", "")
                score = hit["score"]

                if not doc_id or not ObjectId.is_valid(doc_id):
                    continue

                doc = document_map.get(doc_id)
                if doc:
                    dto = SearchDocumentDTO.from_model(model=doc, content=content, score=score)
                    result_dtos.append(dto)

            self.logger.info(f"Đã tạo {len(result_dtos)} DTO từ kết quả retrieval")

            # 10. Post-process với LLM để tổ chức kết quả
            if result_dtos and llm:
                contents = [dto.content for dto in result_dtos]
                prompt = f"""
                Cho câu hỏi: "{query_text}"
                Dựa trên các đoạn văn sau:
                {chr(10).join(f"- {c}" for c in contents)}
                Hãy tổ chức lại thành các phần:
                - Khái niệm
                - Các loại (nếu có)
                - Triệu chứng (nếu có)
                - Nguyên nhân (nếu có)
                Trả về dạng văn bản ngắn gọn, rõ ràng, chỉ bao gồm các phần có thông tin.
                """
                llm_response = await llm.generate(prompt)
                if llm_response:
                    self.logger.info(f"Kết quả tổ chức từ LLM: {llm_response}")
                    llm_score = statistics.mean([dto.score for dto in result_dtos]) if result_dtos else 0.5
                    result_dtos.append(SearchDocumentDTO.from_llm_summary(
                        content=llm_response,
                        score=llm_score
                    ))

            return Result.success(
                data=result_dtos,
                code=RetrievalMessage.FETCHED.code,
                message=RetrievalMessage.FETCHED.message
            )

        except Exception as e:
            self.logger.error(f"Lỗi trong GetRetrievedContextQueryHandler: {e}", exc_info=True)
            return Result.failure(
                code=RetrievalMessage.RETRIEVAL_FAILED.code,
                message=RetrievalMessage.RETRIEVAL_FAILED.message
            )

        finally:
            if self.llm:
                await self.llm.close()