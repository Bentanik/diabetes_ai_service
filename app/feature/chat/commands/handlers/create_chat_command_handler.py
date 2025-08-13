from typing import List, Optional
from app.database import get_collections
from app.database.enums import ChatRoleType
from app.database.models import SettingModel, ChatHistoryModel, ChatSessionModel
from app.dto.models.chat_history_model_dto import ChatHistoryModelDTO
from core.cqrs import CommandRegistry, CommandHandler
from core.llm.gemini import GeminiClient
from core.result import Result
from rag.vector_store import VectorStoreOperations
from rag.vector_store.operations import SearchResult
from shared.messages import ChatResult, SettingResult
from utils import get_logger
from ..create_chat_command import CreateChatCommand


@CommandRegistry.register_handler(CreateChatCommand)
class CreateChatCommandHandler(CommandHandler):
    def __init__(self):
        super().__init__()
        self.logger = get_logger(__name__)
        self.db = get_collections()
        self.vector_operations = VectorStoreOperations.get_instance()

    async def execute(self, command: CreateChatCommand) -> Result[None]:
        # Lấy setting từ Database
        setting = await self.db.settings.find_one({})
        if not setting:
            return Result.failure(
                message=SettingResult.NOT_FOUND.message,
                code=SettingResult.NOT_FOUND.code,
                data=[],
            )
        setting = SettingModel.from_dict(setting)

        # Tạo session
        session_id = await self.create_session(user_id=command.user_id, title=command.content, session_id=command.session_id)

        # Cải thiện query search
        search_enhance = await self.enhance_query(command.content)

        # Lưu data câu hỏi vào trước
        chat_user = ChatHistoryModel(
            session_id=session_id,
            user_id=command.user_id,
            content=command.content,
            role=ChatRoleType.USER
        )

        # Lưu câu hỏi vào database
        await self.save_data(data=chat_user)

        # Tìm kiếm data trong vector store
        search_result = await self.search_data(search=search_enhance, setting=setting)
        
        # Lấy lịch sử trò chuyện từ database
        chat_histories: List[ChatHistoryModel] = []
        if session_id is not None:
           chat_histories = await self.get_histories(session_id=session_id)

        if len(chat_histories) > 0:
            chat_histories.reverse()

        # Đưa vô LLM gen ra với các thông tin như kết quả tìm kiếm, lịch sử trò chuyện
        gen_text = await self.gen_data_with_llm(
            message = search_enhance,
            contexts = search_result,
            setting = setting,
            histories=chat_histories
        )

        # Lưu câu trả lời của AI vào database
        chat_ai = ChatHistoryModel(
            session_id=session_id,
            user_id=command.user_id,
            content=gen_text,
            role=ChatRoleType.AI
        )
        await self.save_data(data=chat_ai)
        
        chat_history_dto = ChatHistoryModelDTO.from_model(chat_ai)

        return Result.success(
            code=ChatResult.CHAT_CREATED.code,
            message=ChatResult.CHAT_CREATED.message,
            data=chat_history_dto,
        )

    async def create_session(self, user_id: str, title: str, session_id: Optional[str]) -> str:
        if(user_id == "admin"):
            return "session_admin"

        # Kiểm tra có session chưa
        if await self.db.chat_sessions.count_documents({
            "session_id": session_id
        }) > 0:
            return session_id
        
        session = ChatSessionModel(
                    user_id=user_id,
                    title=(title[:100] + "..." if len(title) > 100 else title),
                )
        await self.db.chat_sessions.insert_one(session.to_dict())
        return str(session.id)

    async def enhance_query(self, content: str) -> str:
        return content

    async def search_data(self, search: str, setting: SettingModel) -> List[str]:
        vector_operation = VectorStoreOperations.get_instance()
        search_result: List[SearchResult] = await vector_operation.search(
            query_text=search,
            top_k=setting.top_k,
            score_threshold=setting.search_accuracy,
            collection_names=(
                setting.list_knowledge_ids if setting.list_knowledge_ids else None
            ),
        )

        search_texts = [item.text for item in search_result]

        return search_texts

    async def get_histories(self, session_id: str) -> List[ChatHistoryModel]:
        chat_history_cursor = self.db.chat_histories.find(
            {"session_id": session_id}
        ).sort("updated_at", -1).limit(20)

        chat_history_list = await chat_history_cursor.to_list(length=20)

        if not chat_history_list:
            return []

        chat_histories = [ChatHistoryModel.from_dict(doc) for doc in chat_history_list]
        
        return chat_histories
    
    async def gen_data_with_llm(
        self, message: str, contexts: List[str], setting: SettingModel, 
        histories: List[ChatHistoryModel]
    ) -> str:
        llm_client = GeminiClient()
        llm_client.set_temperature(setting.temperature)
        
        chat_histories = [
            {
                "role": x.role.value if hasattr(x.role, "value") else str(x.role),
                "content": x.content
            }
            for x in histories
        ]

        response = llm_client.prompt(
            message=message,
            system_prompt=setting.system_prompt,
            context=contexts,
            history=chat_histories,
        )

        return response

    async def save_data(self, data: ChatHistoryModel) -> bool:
        try:
            await self.db.chat_histories.insert_one(data.to_dict())
            return True
        except:
            return False
    