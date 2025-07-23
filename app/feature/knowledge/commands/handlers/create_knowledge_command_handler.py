from app.database.manager import get_collections
from app.database.models import KnowledgeModel
from core.result import Result
from core.cqrs import CommandRegistry, CommandHandler
from app.feature.knowledge import CreateKnowledgeCommand
from shared.messages import KnowledgeResult
from utils import get_logger


@CommandRegistry.register_handler(CreateKnowledgeCommand)
class CreateKnowledgeCommandHandler(CommandHandler):
    def __init__(self):
        super().__init__()
        self.logger = get_logger(__name__)

    async def execute(self, command: CreateKnowledgeCommand) -> Result[None]:
        """
        Tạo cơ sở tri thức
        Args:
            command: CreateKnowledgeCommand

        Returns:
            Result[None]: Kết quả thành công hoặc lỗi
        """
        self.logger.info(f"Tạo cơ sở tri thức mới: {command.name}")

        collection = get_collections()
        # Kiểm tra tồn tại
        exists = await collection.knowledges.count_documents({"name": command.name}) > 0
        if exists:
            self.logger.warning(f"Tên cơ sở tri thức đã tồn tại: {command.name}")
            return Result.failure(
                message=KnowledgeResult.NAME_EXISTS.message,
                code=KnowledgeResult.NAME_EXISTS.code,
            )

        # Tạo model
        knowledge = KnowledgeModel(name=command.name, description=command.description)

        # Lưu vào database
        await collection.knowledges.insert_one(knowledge.to_dict())

        self.logger.info(f"Cơ sở tri thức đã được tạo thành công: {command.name}")

        return Result.success(
            message=KnowledgeResult.CREATED.message,
            code=KnowledgeResult.CREATED.code,
            data=None,
        )
