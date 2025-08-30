import os
import dotenv
from typing import List, Optional
from datetime import datetime
from bson import ObjectId

from sklearn.metrics.pairwise import cosine_similarity

from core.cqrs import CommandRegistry, CommandHandler
from core.embedding import EmbeddingModel
from core.llm import QwenLLM
from core.result import Result
from rag.vector_store import VectorStoreManager
from shared.messages import ChatMessage, SettingMessage
from app.database import get_collections
from app.database.enums import ChatRoleType
from app.database.models import ChatHistoryModel, ChatSessionModel, SettingModel
from app.dto.models import ChatHistoryModelDTO
from utils import get_logger
from ..create_chat_command import CreateChatCommand
from shared.default_rag_prompt import SYSTEM_PROMPT

dotenv.load_dotenv()

@CommandRegistry.register_handler(CreateChatCommand)
class CreateChatCommandHandler(CommandHandler):
    def __init__(self):
        super().__init__()
        self.logger = get_logger(__name__)
        self.db = get_collections()
        self.vector_store_manager = VectorStoreManager()
        self.embedding_model = EmbeddingModel()
        self.llm_client: Optional[QwenLLM] = None

    async def get_llm_client(self) -> QwenLLM:
        """Lazily initialize LLM client"""
        if self.llm_client is None:
            self.llm_client = QwenLLM(
                model=os.getenv("QWEN_MODEL", "qwen2.5:3b-instruct"),
                base_url=os.getenv("QWEN_URL", "http://localhost:11434")
            )
        return self.llm_client

    async def execute(self, command: CreateChatCommand) -> Result[None]:
        try:
            # 1. Lấy cài đặt hệ thống
            settings_doc = await self.db.settings.find_one({})
            if not settings_doc:
                return Result.failure(
                    code=SettingMessage.NOT_FOUND.code,
                    message=SettingMessage.NOT_FOUND.message
                )
            settings = SettingModel.from_dict(settings_doc)

            # 2. Tạo hoặc lấy session
            session = await self.create_session(
                user_id=command.user_id,
                title=command.content,
                session_id=command.session_id,
                use_external_knowledge=command.use_external_knowledge
            )

            # 3. Lưu tin nhắn người dùng
            user_chat = ChatHistoryModel(
                session_id=session.id,
                user_id=command.user_id,
                content=command.content,
                role=ChatRoleType.USER
            )
            await self.save_data(user_chat)

            # 4. Lấy lịch sử trò chuyện (mới nhất trước)
            histories = await self.get_histories(session_id=session.id)
            histories.reverse()

            # 5. Retrieval: tìm kiếm thông tin liên quan
            context_texts = []
            if settings.list_knowledge_ids and not command.use_external_knowledge:
                enhanced_query = await self.enhance_query(command.content)
                context_texts = await self.search_data(enhanced_query, settings)

            # 6. Sinh câu trả lời
            gen_text = await self.gen_data_with_llm(
                message=command.content,
                contexts=context_texts,
                histories=histories,
                settings=settings,
                use_external_knowledge=command.use_external_knowledge
            )

            # 7. Lưu câu trả lời AI
            ai_chat = ChatHistoryModel(
                session_id=session.id,
                user_id=command.user_id,
                content=gen_text,
                role=ChatRoleType.AI
            )
            await self.save_data(ai_chat)

            # 8. Cập nhật thời gian session
            await self.update_session(session_id=session.id)

            # 9. Trả kết quả
            dto = ChatHistoryModelDTO.from_model(ai_chat)
            return Result.success(
                code=ChatMessage.CHAT_CREATED.code,
                message=ChatMessage.CHAT_CREATED.message,
                data=dto
            )

        except Exception as e:
            self.logger.error(f"Error in CreateChatCommandHandler: {e}", exc_info=True)
            return Result.failure(
                code=ChatMessage.CHAT_ERROR.code,
                message=ChatMessage.CHAT_ERROR.message
            )

    async def create_session(
        self,
        user_id: str,
        title: str,
        session_id: Optional[str],
        use_external_knowledge: bool = False
    ) -> ChatSessionModel:
        """Tạo hoặc lấy session hiện có"""
        # Admin luôn dùng session riêng
        if user_id == "admin":
            chat_session = await self.db.chat_sessions.find_one({"user_id": "admin"})
            if chat_session:
                return ChatSessionModel.from_dict(chat_session)
            session = ChatSessionModel(
                user_id="admin",
                title="Test AI",
                external_knowledge=use_external_knowledge
            )
            await self.db.chat_sessions.insert_one(session.to_dict())
            return session

        # Dùng session_id nếu có
        if session_id:
            chat_session = await self.db.chat_sessions.find_one({"_id": ObjectId(session_id)})
            if chat_session:
                return ChatSessionModel.from_dict(chat_session)

        # Tạo session mới
        session_title = title[:100] + "..." if len(title) > 100 else title
        session = ChatSessionModel(
            user_id=user_id,
            title=session_title,
            external_knowledge=False
        )
        await self.db.chat_sessions.insert_one(session.to_dict())
        return session

    async def update_session(self, session_id: str) -> bool:
        """Cập nhật thời gian cập nhật session"""
        try:
            result = await self.db.chat_sessions.update_one(
                {"_id": ObjectId(session_id)},
                {"$set": {"updated_at": datetime.utcnow()}}
            )
            return result.modified_count > 0
        except Exception as e:
            self.logger.error(f"Error updating session {session_id}: {e}", exc_info=True)
            return False

    async def enhance_query(self, query: str) -> str:
        """Mở rộng câu hỏi để retrieval tốt hơn bằng LLM"""
        if len(query.strip()) < 5:
            return query

        llm = await self.get_llm_client()

        prompt = """
Hãy mở rộng câu hỏi sau thành các câu hỏi con hoặc cụm từ tìm kiếm liên quan giúp tìm thông tin tốt hơn.
Chỉ liệt kê mỗi dòng một cụm, không giải thích.

Ví dụ:
Câu hỏi: "Tác dụng của vitamin C?"
Trả lời:
- Tác dụng của vitamin C
- Vitamin C có lợi gì cho sức khỏe
- Vitamin C hỗ trợ miễn dịch như thế nào
- Liều lượng vitamin C hàng ngày

Câu hỏi: {query}
Trả lời:
""".strip().format(query=query)

        try:
            enhanced = await llm.generate(prompt=prompt, temperature=0.7)
            lines = [line.strip("- ").strip() for line in enhanced.strip().split("\n") if line.strip()]
            expanded = " ".join(lines)
            return f"{query} {expanded}"
        except Exception as e:
            self.logger.warning(f"Query enhancement failed: {e}")
            return query  # fallback

    async def get_histories(self, session_id: str) -> List[ChatHistoryModel]:
        """Lấy lịch sử chat (tối đa 20 tin gần nhất)"""
        cursor = self.db.chat_histories.find({"session_id": session_id}) \
            .sort("updated_at", -1).limit(20)
        docs = await cursor.to_list(length=20)
        return [ChatHistoryModel.from_dict(doc) for doc in docs]

    async def search_data(self, search: str, settings: SettingModel) -> List[str]:
        """Tìm kiếm vector trong các knowledge collections"""
        try:
            text_embedding = await self.embedding_model.embed(search)
            results = await self.vector_store_manager.search_async(
                collections=settings.list_knowledge_ids,
                query_vector=text_embedding,
                top_k=settings.top_k * 2,
                score_threshold=settings.search_accuracy
            )
            contents = []
            for hits in results.values():
                for hit in hits:
                    content = hit["payload"].get("content")
                    if content:
                        contents.append(content)
            return contents
        except Exception as e:
            self.logger.error(f"Vector search error: {e}", exc_info=True)
            return []

    async def filter_relevant_contexts(self, question: str, contexts: List[str], threshold: float = 0.55) -> List[str]:
        """Lọc context chỉ giữ lại những đoạn liên quan đến câu hỏi"""
        if not contexts:
            return []
        try:
            embeddings = await self.embedding_model.embed_batch([question] + contexts)
            question_emb = embeddings[0]
            context_embs = embeddings[1:]
            sims = cosine_similarity([question_emb], context_embs)[0]
            return [ctx for ctx, sim in zip(contexts, sims) if sim >= threshold]
        except Exception as e:
            self.logger.warning(f"Context filtering failed: {e}", exc_info=True)
            return contexts

    async def gen_data_with_llm(
        self,
        message: str,
        contexts: List[str],
        histories: List[ChatHistoryModel],
        settings: SettingModel,
        use_external_knowledge: bool
    ) -> str:
        """Sinh câu trả lời bằng LLM với context và lịch sử"""
        llm = await self.get_llm_client()

        if use_external_knowledge:
            system_prompt = f"{SYSTEM_PROMPT}\nBạn có thể sử dụng kiến thức chung để trả lời."
            context_str = ""
        else:
            # Lọc context chỉ giữ phần liên quan
            filtered_contexts = await self.filter_relevant_contexts(message, contexts)
            selected_contexts = filtered_contexts[:settings.top_k]

            if not selected_contexts:
                return "Không tìm thấy thông tin liên quan. Bạn có muốn hỏi với kiến thức ngoài không?"

            # Làm sạch thẻ và định dạng
            cleaned_contexts = []
            for i, ctx in enumerate(selected_contexts):
                ctx = ctx.replace("[HEADING]", "### ").replace("[/HEADING]", "")
                ctx = ctx.replace("[SUBHEADING]", "#### ").replace("[/SUBHEADING]", "")
                cleaned_contexts.append(f"[Tài liệu {i+1}]\n{ctx}")

            context_str = "\n\n".join(cleaned_contexts)
            system_prompt = f"{SYSTEM_PROMPT}\n\n### Thông tin tham khảo:\n{context_str}"

        # Xây dựng prompt theo ChatML
        prompt_lines = [f"<|im_start|>system\n{system_prompt}<|im_end|>"]
        for msg in histories:
            role = "assistant" if msg.role == ChatRoleType.AI else "user"
            prompt_lines.append(f"<|im_start|>{role}\n{msg.content}<|im_end|>")
        prompt_lines.append(f"<|im_start|>user\n{message}<|im_end|>")
        prompt_lines.append("<|im_start|>assistant")

        final_prompt = "\n".join(prompt_lines)

        try:
            response = await llm.generate(prompt=final_prompt, temperature=settings.temperature)
            return response.strip()
        except Exception as e:
            self.logger.error(f"LLM generation error: {e}", exc_info=True)
            return "Xin lỗi, đã xảy ra lỗi khi tạo câu trả lời."

    async def save_data(self, data: ChatHistoryModel) -> bool:
        """Lưu tin nhắn vào DB"""
        try:
            result = await self.db.chat_histories.insert_one(data.to_dict())
            return result.acknowledged
        except Exception as e:
            self.logger.error(f"Error saving chat history: {e}", exc_info=True)
            return False