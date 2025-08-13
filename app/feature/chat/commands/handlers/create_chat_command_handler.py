from typing import List
from app.database import get_collections
from app.database.models import SettingModel
from core.cqrs import CommandRegistry, CommandHandler
from core.llm.gemini import GeminiConfig, GeminiClient, GeminiManager
from core.result import Result
from rag.vector_store import VectorStoreOperations
from rag.vector_store.operations import SearchResult
from shared.messages import ChatResult, SettingResult
from bson import ObjectId
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

        setting = await self.db.settings.find_one({})
        if not setting:
            return Result.failure(
                    message=SettingResult.NOT_FOUND.message,
                    code=SettingResult.NOT_FOUND.code,
                    data=[],
                )
        setting = SettingModel.from_dict(setting)
        
        search_enhance = await self.enhance_query(command.content)

        search_result = await self.search_data(search=search_enhance, setting = setting)

        return Result.success(
            code=ChatResult.CHAT_CREATED.code,
            message=ChatResult.CHAT_CREATED.message,
            data=search_result
        )
    
    async def enhance_query(self, content: str) -> str:
        return content
    
    async def search_data(self, search: str, setting: SettingModel) -> List[str]:
        vector_operation = VectorStoreOperations.get_instance()
        search_result: List[SearchResult] = await vector_operation.search(
            query_text=search,
            top_k = setting.top_k,
            score_threshold=setting.search_accuracy,
            collection_names=setting.list_knowledge_ids if setting.list_knowledge_ids else None
        )

        search_texts = [item.text for item in search_result]


        return search_texts
    
    async def gen_data_with_llm(self, contents: List[str], setting: SettingModel) -> str:
        llm_config = GeminiConfig(
            temperature=setting.temperature,
        )
        llm_client = GeminiClient(llm_config)
        if not llm_client.connect(): 
            return
        
        llm_manager = GeminiManager(
            client=llm_client
        )

        llm_manager.initialize_conversation(setting.system_prompt, )


