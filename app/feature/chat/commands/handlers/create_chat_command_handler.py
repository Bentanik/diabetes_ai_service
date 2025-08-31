import os
import dotenv
from typing import List
from datetime import datetime
from bson import ObjectId

from core.cqrs import CommandRegistry, CommandHandler
from core.embedding import EmbeddingModel
from core.llm import QwenLLM
from core.result import Result
from rag.vector_store import VectorStoreManager
from rag.retrieval.retriever import Retriever
from shared.messages import ChatMessage, SettingMessage
from app.database import get_collections
from app.database.enums import ChatRoleType
from app.database.models import ChatHistoryModel, ChatSessionModel, SettingModel
from app.dto.models import ChatHistoryModelDTO
from utils import get_logger
from ..create_chat_command import CreateChatCommand
from shared.rag_templates import render_template

dotenv.load_dotenv()


@CommandRegistry.register_handler(CreateChatCommand)
class CreateChatCommandHandler(CommandHandler):
    def __init__(self):
        super().__init__()
        self.logger = get_logger(__name__)
        self.db = get_collections()
        self.vector_store_manager = VectorStoreManager()
        self.embedding_model = None
        self.llm_client = None
        self.retriever_cache = {}

    async def get_llm_client(self) -> QwenLLM:
        """Khởi tạo LLM client (lazy)"""
        if self.llm_client is None:
            self.llm_client = QwenLLM(
                model=os.getenv("QWEN_MODEL", "qwen2.5:3b-instruct"),
                base_url=os.getenv("QWEN_URL", "http://localhost:11434")
            )
        return self.llm_client

    async def get_embedding_model(self) -> EmbeddingModel:
        """Khởi tạo embedding model (lazy)"""
        if self.embedding_model is None:
            self.embedding_model = EmbeddingModel()
        return self.embedding_model

    def get_retriever(self, collections: List[str]) -> Retriever:
        """Cache retriever theo danh sách collections"""
        key = ",".join(sorted(collections))
        if key not in self.retriever_cache:
            self.retriever_cache[key] = Retriever(
                collections=collections,
                vector_store_manager=self.vector_store_manager
            )
        return self.retriever_cache[key]

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
                session_id=command.session_id
            )

            # 3. Lưu tin nhắn người dùng
            user_chat = ChatHistoryModel(
                session_id=session.id,
                user_id=command.user_id,
                content=command.content,
                role=ChatRoleType.USER
            )
            await self.save_data(user_chat)

            # 4. Lấy lịch sử trò chuyện (mới nhất ở dưới)
            histories = await self.get_histories(session_id=session.id)
            histories.reverse()  # để tin mới nhất ở cuối

            # 5. Retrieval (tối ưu)
            context_texts = []
            if settings.list_knowledge_ids:
                try:
                    # Viết lại query nếu có tham chiếu
                    rewritten_query = await self.rewrite_query_if_needed(command.content, histories)

                    # Embed query
                    embedding_model = await self.get_embedding_model()
                    query_vector = await embedding_model.embed(rewritten_query)

                    # Dùng retriever đã cache
                    retriever = self.get_retriever(settings.list_knowledge_ids)
                    raw_results = await retriever.retrieve(
                        query_vector=query_vector,
                        top_k=settings.top_k * 2
                    )

                    # Lọc theo score từ retriever
                    score_threshold = getattr(settings, "search_accuracy", 0.5)
                    filtered_results = [
                        hit for hit in raw_results
                        if hit["score"] >= score_threshold
                    ]

                    # Lấy nội dung
                    context_texts = [
                        hit["payload"]["content"]
                        for hit in filtered_results
                        if hit["payload"] and hit["payload"].get("content")
                    ]
                    context_texts = context_texts[:settings.top_k]

                    self.logger.info(f"Retrieved {len(context_texts)} contexts for: '{rewritten_query}'")

                except Exception as e:
                    self.logger.error(f"Retrieval failed: {e}", exc_info=True)

            # 6. Sinh câu trả lời tự nhiên, có Markdown
            gen_text = await self.gen_natural_response(
                message=command.content,
                contexts=context_texts,
                histories=histories,
                settings=settings
            )

            # 7. Lưu câu trả lời AI
            ai_chat = ChatHistoryModel(
                session_id=session.id,
                user_id=command.user_id,
                content=gen_text,
                role=ChatRoleType.AI
            )
            await self.save_data(ai_chat)

            # 8. Cập nhật thời gian phiên
            await self.update_session(session_id=session.id)

            # 9. Trả về DTO thành công
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
        session_id: str = None
    ) -> ChatSessionModel:
        """Tạo hoặc lấy session"""
        if user_id == "admin":
            doc = await self.db.chat_sessions.find_one({"user_id": "admin"})
            if doc:
                return ChatSessionModel.from_dict(doc)
            session = ChatSessionModel(user_id="admin", title="Test AI")
            await self.db.chat_sessions.insert_one(session.to_dict())
            return session

        if session_id:
            doc = await self.db.chat_sessions.find_one({"_id": ObjectId(session_id)})
            if doc:
                return ChatSessionModel.from_dict(doc)

        session_title = title[:100] + "..." if len(title) > 100 else title
        session = ChatSessionModel(
            user_id=user_id,
            title=session_title
        )
        await self.db.chat_sessions.insert_one(session.to_dict())
        return session

    async def update_session(self, session_id: str) -> bool:
        """Cập nhật updated_at"""
        try:
            result = await self.db.chat_sessions.update_one(
                {"_id": ObjectId(session_id)},
                {"$set": {"updated_at": datetime.utcnow()}}
            )
            return result.modified_count > 0
        except Exception as e:
            self.logger.error(f"Update session failed: {e}", exc_info=True)
            return False

    async def get_histories(self, session_id: str) -> List[ChatHistoryModel]:
        """Lấy 20 tin nhắn gần nhất"""
        cursor = self.db.chat_histories.find({"session_id": session_id}) \
            .sort("updated_at", -1).limit(20)
        docs = await cursor.to_list(length=20)
        histories = []
        for doc in docs:
            model = ChatHistoryModel.from_dict(doc)
            if isinstance(model.role, str):
                model.role = ChatRoleType.USER if model.role.lower() == "user" else ChatRoleType.AI
            histories.append(model)
        return histories

    async def save_data(self, data: ChatHistoryModel) -> bool:
        """Lưu tin nhắn vào DB"""
        try:
            result = await self.db.chat_histories.insert_one(data.to_dict())
            return result.acknowledged
        except Exception as e:
            self.logger.error(f"Save chat history failed: {e}", exc_info=True)
            return False

    async def rewrite_query_if_needed(self, query: str, histories: List[ChatHistoryModel]) -> str:
        """Viết lại query nếu có từ tham chiếu (rule-based)"""
        if len(histories) < 2:
            return query

        pronouns = ["nó", "vậy", "đó", "kia", "trên", "dưới", "trước", "sau", "ý", "cái"]
        if not any(p in query.lower() for p in pronouns):
            return query

        for msg in reversed(histories):
            if msg.role == ChatRoleType.AI:
                content = msg.content.lower()
                if "đái tháo đường" in content:
                    query = query.replace("nó", "đái tháo đường").replace("đó", "đái tháo đường")
                elif "insulin" in content:
                    query = query.replace("nó", "insulin")
                elif "chế độ ăn" in content:
                    query = query.replace("nó", "chế độ ăn")
                break
        return query

    async def gen_natural_response(
        self,
        message: str,
        contexts: List[str],
        histories: List[ChatHistoryModel],
        settings: SettingModel
    ) -> str:
        """Sinh câu trả lời tự nhiên, mềm dẻo, dưới dạng Markdown"""
        llm = await self.get_llm_client()

        if not contexts:
            return "Tôi không tìm thấy thông tin liên quan trong tài liệu để trả lời câu hỏi này."

        # Làm sạch context
        cleaned_contexts = "\n\n".join([
            ctx.strip()
            .replace("[HEADING]", "### ").replace("[/HEADING]", "\n")
            .replace("[SUBHEADING]", "#### ").replace("[/SUBHEADING]", "\n")
            for ctx in contexts if ctx.strip()
        ])

        # Đọc system prompt
        try:
            with open("shared/rag_templates/system_prompt.txt", "r", encoding="utf-8") as f:
                system_prompt = f.read().strip()
        except Exception as e:
            self.logger.warning(f"Không thể đọc system_prompt: {e}")
            system_prompt = "Bạn là chuyên gia y tế, trả lời rõ ràng, dùng Markdown."

        # Render template
        try:
            prompt_text = render_template(
                template_name="response.j2",
                system_prompt=system_prompt,
                contexts=cleaned_contexts,
                question=message
            )
        except Exception as e:
            self.logger.error(f"Render template thất bại: {e}")
            return "Xin lỗi, đã xảy ra lỗi khi tạo câu trả lời."

        # Xây dựng prompt ChatML
        prompt_lines = [f"<|im_start|>system\n{prompt_text}<|im_end|>"]
        for msg in histories:
            role = "assistant" if getattr(msg.role, 'value', msg.role) == "ai" else "user"
            prompt_lines.append(f"<|im_start|>{role}\n{msg.content}<|im_end|>")
        prompt_lines.append(f"<|im_start|>user\n{message}<|im_end|>")
        prompt_lines.append("<|im_start|>assistant")  # đúng cú pháp

        final_prompt = "\n".join(prompt_lines)

        try:
            response = await llm.generate(
                prompt=final_prompt,
                temperature=0.75,
                max_tokens=1800,
                top_p=0.9
            )
            text = response.strip()

            # Làm sạch đầu ra, đảm bảo Markdown
            text = self._clean_and_ensure_markdown(text)
            return text if text else "Không thể tạo câu trả lời."
        except Exception as e:
            self.logger.error(f"LLM generation failed: {e}", exc_info=True)
            return "Xin lỗi, đã xảy ra lỗi khi tạo câu trả lời."

    def _clean_and_ensure_markdown(self, text: str) -> str:
        """Làm sạch và đảm bảo định dạng Markdown hợp lệ"""
        if not text.strip():
            return text

        import re

        # Chuẩn hóa in đậm
        text = re.sub(r'\*\*(.*?)\*\*', r'**\1**', text)
        text = re.sub(r'__(.*?)__', r'**\1**', text)

        # Chuẩn hóa in nghiêng
        text = re.sub(r'\*(.*?)\*', r'*\1*', text)
        text = re.sub(r'_(.*?)_', r'*\1*', text)

        # Đảm bảo tiêu đề có xuống dòng
        text = re.sub(r'(^|\n)(#{1,6} )', r'\1\n\2', text)

        # Loại bỏ phần leak
        lines = text.split('\n')
        cleaned = []
        in_leak = False
        leak_keywords = [
            "hãy suy nghĩ", "phân tích câu hỏi", "trích xuất", "suy luận",
            "gợi ý mở đầu", "gợi ý kết thúc", "trả lời theo cấu trúc"
        ]

        for line in lines:
            lower_line = line.lower()
            if any(kw in lower_line for kw in leak_keywords):
                in_leak = True
                continue
            if line.startswith("### ") and in_leak:
                in_leak = False
                cleaned.append(line)
            elif not in_leak and line.strip():
                cleaned.append(line)

        return '\n'.join(cleaned).strip()