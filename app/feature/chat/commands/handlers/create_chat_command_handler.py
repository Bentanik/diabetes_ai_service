from typing import List, Optional

from datetime import datetime
from bson import ObjectId
from app.database import get_collections
from app.database.enums import ChatRoleType
from app.database.models import ChatHistoryModel, ChatSessionModel, SettingModel
from app.dto.models import ChatHistoryModelDTO
from core.cqrs import CommandRegistry, CommandHandler
from core.embedding import EmbeddingModel
from core.llm import OpenRouterLLM
from core.result import Result
from rag.vector_store import VectorStoreManager
from shared.messages import ChatMessage
from utils import get_logger
from ..create_chat_command import CreateChatCommand
from shared.default_rag_prompt import QA_PROMPT, EXTERNAL_QA_PROMPT, SYSTEM_PROMPT


@CommandRegistry.register_handler(CreateChatCommand)
class CreateChatCommandHandler(CommandHandler):
    def __init__(self):
        super().__init__()
        self.logger = get_logger(__name__)
        self.db = get_collections()
        self.vector_store_manager = VectorStoreManager()
        self.embedding_model = EmbeddingModel()
        self.llm_client = OpenRouterLLM(
            qa_prompt=QA_PROMPT,
            external_qa_prompt=EXTERNAL_QA_PROMPT,
            system_prompt=SYSTEM_PROMPT,
        )

    async def execute(self, command: CreateChatCommand) -> Result[None]:
        # Lấy setting từ database
        settings = await self.db.settings.find_one({})

        settings = SettingModel.from_dict(settings)

        # Tạo session
        session = await self.create_session(user_id=command.user_id, title=command.content, session_id=command.session_id, use_external_knowledge=command.use_external_knowledge)

        # Cải thiện query search
        search_enhance = await self.enhance_query(command.content)

        # Lưu data câu hỏi vào trước
        chat_user = ChatHistoryModel(
            session_id=session.id,
            user_id=command.user_id,
            content=command.content,
            role=ChatRoleType.USER
        )

        # Lưu câu hỏi vào database
        await self.save_data(data=chat_user)
        
        # Lấy lịch sử trò chuyện từ database
        chat_histories: List[ChatHistoryModel] = []
        if session is not None:
           chat_histories = await self.get_histories(session_id=session.id)

        if len(chat_histories) > 0:
            chat_histories.reverse()

        # Tìm kiếm data trong vector store
        search_result = await self.search_data(search=search_enhance, settings=settings)

        # Đưa vô LLM gen ra với các thông tin như kết quả tìm kiếm, lịch sử trò chuyện
        gen_text = await self.gen_data_with_llm(
            message = search_enhance,
            contexts = search_result,
            histories=chat_histories,
            use_external_knowledge=command.use_external_knowledge
        )

        # Lưu câu trả lời của AI vào database
        chat_ai = ChatHistoryModel(
            session_id=session.id,
            user_id=command.user_id,
            content=gen_text,
            role=ChatRoleType.AI
        )
        await self.save_data(data=chat_ai)

        # Cập nhật thời gian của session
        await self.update_session(session_id=session.id)

        chat_history_dto = ChatHistoryModelDTO.from_model(chat_ai)

        return Result.success(
            code=ChatMessage.CHAT_CREATED.code,
            message=ChatMessage.CHAT_CREATED.message,
            data=chat_history_dto,
        )

    async def create_session(self, user_id: str, title: str, session_id: Optional[str], use_external_knowledge: Optional[bool] = False) -> ChatSessionModel:
        # Trường hợp admin
        if user_id == "admin":
            chat_session = await self.db.chat_sessions.find_one({"user_id": user_id})
            if chat_session:
                return ChatSessionModel.from_dict(chat_session)
            # Tạo mới
            session = ChatSessionModel(
                user_id="admin",
                title="Test AI",
                external_knowledge=use_external_knowledge
            )
            await self.db.chat_sessions.insert_one(session.to_dict())
            return session

        # Trường hợp session_id được truyền
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
        """Update session với thời gian mới nhất"""
        try:
            await self.db.chat_sessions.update_one(
                {"_id": ObjectId(session_id)},
                {
                    "$set": {
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            return True
        except Exception as e:
            self.logger.error(f"Error updating session {session_id}: {str(e)}")
            return False

    async def enhance_query(self, content: str) -> str:
        return content

    async def get_histories(self, session_id: str) -> List[ChatHistoryModel]:
        chat_history_cursor = self.db.chat_histories.find(
            {"session_id": session_id}
        ).sort("updated_at", -1).limit(20)

        chat_history_list = await chat_history_cursor.to_list(length=20)

        if not chat_history_list:
            return []

        chat_histories = [ChatHistoryModel.from_dict(doc) for doc in chat_history_list]
        
        return chat_histories
    
    async def search_data(self, search: str, settings: SettingModel) -> List[str]:
        text_embedding = await self.embedding_model.embed(search)

        search_results = await self.vector_store_manager.search_async(
            collections=settings.list_knowledge_ids,
            query_vector=text_embedding,
            top_k=settings.top_k,
            search_accuracy=settings.search_accuracy
        )

        searchs = []
        for items in search_results.values():
            for item in items:
                searchs.append(item["payload"]["content"])

        return searchs
    
    async def gen_data_with_llm(
        self,
        message: str,
        contexts: List[str],
        histories: Optional[List[ChatHistoryModel]] = None,
        use_external_knowledge: Optional[bool] = False
    ) -> str:

        if use_external_knowledge == False and len(contexts) == 0:
            return "Không tìm thấy trong tài liệu. Bạn có muốn hỏi lại với kiến thức ngoài không?"

        context_str = "\n".join(contexts)
        prompt = EXTERNAL_QA_PROMPT if use_external_knowledge is True else QA_PROMPT

        # Chuyển histories sang dict với role + content
        history_dicts = []
        if histories:
            for msg in histories:
                mapped_role = "assistant" if msg.role == ChatRoleType.AI else "user"

                history_dicts.append({
                    "role": mapped_role,
                    "content": msg.content
                })

        chat = await self.llm_client.chat_async(
            question=message,
            context=context_str,
            prompt_template=prompt,
            history=history_dicts
        )

        not_found_keywords = [
                "thông tin trong tài liệu không đủ để trả lời câu hỏi này một cách đầy đủ và chính xác",
                "tôi chỉ có thể dựa trên những gì được ghi trong tài liệu"
            ]
        
        if any(keyword in chat.lower() for keyword in not_found_keywords):
            return "Không tìm thấy trong tài liệu. Bạn có muốn hỏi lại với kiến thức ngoài không?"

        return chat

    async def save_data(self, data: ChatHistoryModel) -> bool:
        try:
            await self.db.chat_histories.insert_one(data.to_dict())
            return True
        except:
            return False
    