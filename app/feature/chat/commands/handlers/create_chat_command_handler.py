import os
import dotenv

from typing import List, Optional
from datetime import datetime
from bson import ObjectId
from app.database import get_collections
from app.database.enums import ChatRoleType
from app.database.models import ChatHistoryModel, ChatSessionModel, SettingModel
from app.dto.models import ChatHistoryModelDTO
from core.cqrs import CommandRegistry, CommandHandler
from core.embedding import EmbeddingModel
from core.llm import QwenLLM
from core.result import Result
from rag.vector_store import VectorStoreManager
from shared.messages import ChatMessage, SettingMessage
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
        self.llm_client = None

    async def get_llm_client(self):
        if self.llm_client is None:
            self.llm_client = QwenLLM(
                model=os.getenv("QWEN_MODEL"),
                base_url=os.getenv("QWEN_URL")
            )
        return self.llm_client

    async def execute(self, command: CreateChatCommand) -> Result[None]:
        try:
            # 1. Lấy setting
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

            # 3. Lưu câu hỏi của người dùng
            user_chat = ChatHistoryModel(
                session_id=session.id,
                user_id=command.user_id,
                content=command.content,
                role=ChatRoleType.USER
            )
            await self.save_data(user_chat)

            # 4. Lấy lịch sử hội thoại
            histories = await self.get_histories(session_id=session.id)
            histories.reverse()

            # 5. Retrieval
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

            # 7. Lưu câu trả lời của AI
            ai_chat = ChatHistoryModel(
                session_id=session.id,
                user_id=command.user_id,
                content=gen_text,
                role=ChatRoleType.AI
            )
            await self.save_data(ai_chat)

            # 8. Cập nhật session
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
        # Admin case
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
        session = ChatSessionModel(
            user_id=user_id,
            title=(title[:100] + "..." if len(title) > 100 else title),
            external_knowledge=False
        )
        await self.db.chat_sessions.insert_one(session.to_dict())
        return session

    async def update_session(self, session_id: str) -> bool:
        try:
            await self.db.chat_sessions.update_one(
                {"_id": ObjectId(session_id)},
                {"$set": {"updated_at": datetime.utcnow()}}
            )
            return True
        except Exception as e:
            self.logger.error(f"Error updating session {session_id}: {e}")
            return False

    async def enhance_query(self, content: str) -> str:
        return content

    async def get_histories(self, session_id: str) -> List[ChatHistoryModel]:
        cursor = self.db.chat_histories.find({"session_id": session_id}).sort("updated_at", -1).limit(20)
        docs = await cursor.to_list(length=20)
        return [ChatHistoryModel.from_dict(doc) for doc in docs]

    async def search_data(self, search: str, settings: SettingModel) -> List[str]:
        text_embedding = await self.embedding_model.embed(search)
        results = await self.vector_store_manager.search_async(
            collections=settings.list_knowledge_ids,
            query_vector=text_embedding,
            top_k=settings.top_k,
            score_threshold=settings.search_accuracy
        )
        contents = []
        for hits in results.values():
            for hit in hits:
                content = hit["payload"].get("content")
                if content:
                    contents.append(content)
        return contents

    async def gen_data_with_llm(
        self,
        message: str,
        contexts: List[str],
        histories: List[ChatHistoryModel],
        settings: SettingModel,
        use_external_knowledge: bool
    ) -> str:
        llm = await self.get_llm_client()

        if not contexts and not use_external_knowledge:
            return "Không tìm thấy thông tin liên quan. Bạn có muốn hỏi với kiến thức ngoài không?"

        prompt_lines = []

        system_prompt = SYSTEM_PROMPT
        if use_external_knowledge:
            system_prompt += "\nBạn có thể sử dụng kiến thức chung để trả lời."
        else:
            top_k = settings.top_k
            selected_contexts = contexts[:top_k]
            if not selected_contexts:
                return "Không có thông tin liên quan để trả lời."

            cleaned_contexts = []
            for i, ctx in enumerate(selected_contexts):
                ctx = ctx.replace("[HEADING]", "### ").replace("[/HEADING]", "")
                cleaned_contexts.append(f"[Tài liệu {i+1}]\n{ctx}")

            context_str = "\n\n".join(cleaned_contexts)
            system_prompt += f"\nThông tin tham khảo:\n{context_str}"

        prompt_lines.append(f"<|im_start|>system\n{system_prompt}<|im_end|>")

        for msg in histories:
            role = "assistant" if msg.role == ChatRoleType.AI else "user"
            prompt_lines.append(f"<|im_start|>{role}\n{msg.content}<|im_end|>")

        prompt_lines.append(f"<|im_start|>user\n{message}<|im_end|>")
        prompt_lines.append("<|im_start|>assistant")

        final_prompt = "\n".join(prompt_lines)

        try:
            response = await llm.generate(prompt=final_prompt, temperature=settings.temperature)
            return response
        except Exception as e:
            self.logger.error(f"LLM generation error: {e}")
            return "Xin lỗi, đã xảy ra lỗi khi tạo câu trả lời."

    async def save_data(self, data: ChatHistoryModel) -> bool:
        try:
            await self.db.chat_histories.insert_one(data.to_dict())
            return True
        except Exception as e:
            self.logger.error(f"Error saving chat history: {e}")
            return False