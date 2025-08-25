from bson import ObjectId
from pymongo import ReturnDocument
from app.database.manager import get_collections
from core.cqrs import CommandHandler, CommandRegistry
from core.result import Result
from rag.vector_store import VectorStoreManager
from shared.messages.document_message import DocumentMessage
from utils import get_logger
from ..update_document_command import UpdateDocumentCommand


@CommandRegistry.register_handler(UpdateDocumentCommand)
class UpdateDocumentCommandHandler(CommandHandler):
    def __init__(self):
        super().__init__()
        self.db = get_collections()
        self.vector_store_manager = VectorStoreManager()
        self.logger = get_logger(__name__)

    async def execute(self, command: UpdateDocumentCommand) -> Result[None]:
        try:
            document_id = ObjectId(command.document_id)
            update_pipeline = []

            if command.title is not None:
                document_title_exists = await self.db.documents.count_documents({"title": command.title})
                if document_title_exists > 0:
                    return Result.failure(message=DocumentMessage.TITLE_EXISTS.message, code=DocumentMessage.TITLE_EXISTS.code)

                update_pipeline.append({"$set": {"title": command.title}})

            if command.description is not None:
                update_pipeline.append({"$set": {"description": command.description}})

            if command.is_active is not None:
                update_pipeline.append({"$set": {"is_active": command.is_active}})

            updated_doc = await self.db.documents.find_one_and_update(
                {"_id": document_id},
                update_pipeline,
                return_document=ReturnDocument.AFTER
            )

            if command.is_active is not None:
                await self.vector_store_manager.update_payload_async(
                collection_name=str(updated_doc["knowledge_id"]),
                payload_updates={
                    "document_is_active": updated_doc["is_active"],
                },
                metadata__document_id=str(updated_doc["_id"])
            )

            return Result.success(message=DocumentMessage.UPDATED.message, code=DocumentMessage.UPDATED.code)

        except Exception as e:
            self.logger.error(f"Lỗi khi thay đổi trạng thái tài liệu: {e}", exc_info=True)
            return Result.failure(f"Lỗi xử lý: {str(e)}", "error_processing")
