"""
Chat Query Handler - Xử lý truy vấn lấy thông tin chat

File này định nghĩa handler để xử lý ChatQuery, thực hiện việc
lấy thông tin chi tiết của một chat từ database dựa trên session_id.
"""

from bson import ObjectId
from app.database import get_collections
from app.database.models import ChatModel
from app.dto.models import ChatModelDTO
from ..chat_query import ChatQuery
from core.cqrs import QueryHandler, QueryRegistry
from core.result import Result
from shared.messages import ChatResult
from utils import get_logger


@QueryRegistry.register_handler(ChatQuery)
class ChatQueryHandler(QueryHandler[Result[ChatModelDTO]]):
    """
    Handler xử lý truy vấn ChatQuery để lấy thông tin chat.
    """

    def __init__(self):
        """
        Khởi tạo handler
        """
        super().__init__()
        self.logger = get_logger(__name__)

    async def execute(self, query: ChatQuery) -> Result[ChatModelDTO]:
        """
        Thực thi truy vấn lấy thông tin chat theo session_id, truy xuất context và sinh phản hồi bằng Gemini LLM.
        """

        from app.database.manager import get_collections, connect_to_mongodb, close_mongodb_connection
        try:
            await connect_to_mongodb()
            self.logger.info(f"Lấy chat theo session_id: {query.session_id}")

            # Kiểm tra tính hợp lệ của session_id (nếu cần)
            if not query.session_id or not isinstance(query.session_id, str):
                return Result.failure(message="session_id không hợp lệ", code="invalid_session_id")

            # Truy vấn database
            collection = get_collections()
            doc = await collection.chats.find_one({"session_id": query.session_id})
            


            if not doc:
                # Nếu chưa có session_id, dùng retriever với query.content (câu hỏi đầu tiên)
                from rag.retriever import Retriever
                retriever = Retriever()
                user_query = query.content.strip() if query.content else ""
                context_results = await retriever.retrieve(query=user_query, top_k=30, rerank_top_n=6)
                context_texts = [r.text for r in context_results]

                from core.llm.load_llm.gemini import get_gemini_llm
                llm = get_gemini_llm()
                context_str = "\n".join(context_texts)
                prompt = (
                    f"You are a medical expert specializing in diabetes. "
                    f"Explain the following question in Vietnamese clearly, accurately, and comprehensively, but keep your answer concise within 3 to 5 sentences. "
                    f"Use correct medical terms, but explain them simply so that non-experts can understand. "
                    f"Focus on the essential aspects of diabetes relevant to the question. "
                    f"Do not include unrelated information about other diseases or treatments unless specifically asked.\n\n"
                    f"Question: '{user_query}'\n\n"
                    f"Context:\n{context_str}"
                )
                llm_response = llm.invoke(prompt).content

                chat_model = ChatModel(
                    session_id=query.session_id,
                    user_id=query.user_id,
                    content=user_query,
                    context=context_texts,
                    response=llm_response
                )
            
                await collection.chats.insert_one(chat_model.to_dict())

                return Result.success(
                    message=ChatResult.CREATED.message,
                    code=ChatResult.CREATED.code,
                    data={
                        "retrieved_context": context_texts,
                        "llm_response": llm_response,
                    },
                )

            # Chuyển sang model và DTO
            chat_model = ChatModel.from_dict(doc)
            dto = ChatModelDTO.from_model(chat_model)

            # --- Truy xuất context bằng retriever ---
            from rag.retriever import Retriever
            retriever = Retriever()
            # Lấy câu hỏi cuối cùng từ content (giả sử content là lịch sử chat dạng text)
            user_query = chat_model.content.strip().split('\n')[-1] if chat_model.content else ""
            context_results = await retriever.retrieve(query=user_query, top_k=30, rerank_top_n=6)
            context_texts = [r.text for r in context_results]

            # --- Sinh phản hồi bằng Gemini LLM ---
            from core.llm.load_llm.gemini import get_gemini_llm
            llm = get_gemini_llm()
            context_str = "\n".join(context_texts)
            prompt = (
                f"Answer the following question in Vietnamese. Your response must be clear, complete, and easy to understand, written in 3 to 5 full sentences. "
                f"Use only the information provided in the context. Do not include unrelated details or elaborate on disease types, treatments, or complications unless the question asks about them. "
                f"Focus on making the explanation natural and informative enough for someone unfamiliar with the topic to understand clearly.\n\n"
                f"Question: '{user_query}'\n\n"
                f"Context:\n{context_str}"
            )
            llm_response = llm.invoke(prompt).content

            # Bổ sung context và response vào DTO (nếu muốn trả về cho client)
            chat_model = ChatModel(
                session_id=query.session_id,
                user_id=query.user_id,
                content=user_query,
                context=context_texts,
                llm_response=llm_response
            )

            dto_dict = dto.dict()
            dto_dict["retrieved_context"] = context_texts
            dto_dict["llm_response"] = llm_response
            
            await collection.chats.insert_one(chat_model.to_dict())

            return Result.success(
                message=ChatResult.CREATED.message,
                code=ChatResult.CREATED.code,
                data=dto_dict,
            )

        except Exception as e:
            self.logger.error(f"Lỗi khi lấy chat theo session_id: {e}", exc_info=True)
            return Result.failure(message="Lỗi hệ thống", code="error")
        
        finally:
            await close_mongodb_connection()

if __name__ == "__main__":
    import asyncio
    from app.feature.chat.queries import ChatQuery

    async def test_chat_query_handler():
        handler = ChatQueryHandler()
        query = ChatQuery(session_id="test-session-id", user_id="60c72b2f9b1e8b001c8e4d3a", content="Bệnh tiểu đường là gì?")
        result = await handler.execute(query)
        print(result)

    asyncio.run(test_chat_query_handler())