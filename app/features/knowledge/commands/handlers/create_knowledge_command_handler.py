from core.result import Result
from core.cqrs.base import CommandHandler
from core.cqrs.command_registry import CommandRegistry
from app.features.knowledge import CreateKnowledgeCommand
from utils import get_logger


@CommandRegistry.register_handler(CreateKnowledgeCommand)
class CreateKnowledgeCommandHandler(CommandHandler):
    def __init__(self):
        super().__init__()
        self.logger = get_logger(__name__)

    async def execute(self, command: CreateKnowledgeCommand) -> Result[None]:
        try:
            self.logger.info(f"Tạo thư mục: {command.name}")

            self.logger.info(f"Thư mục đã được tạo thành công: {command.name}")
            return Result.success(
                message="Thư mục đã được tạo thành công",
                code="KNOWLEDGE_CREATED",
                data=None,
            )
        except ValueError as e:
            self.logger.error(f"Validation error: {str(e)}")
            return Result.validation_error(message=str(e), code="VALIDATION_ERROR")
        except Exception as e:
            self.logger.error(f"Unexpected error creating knowledge: {str(e)}")
            return Result.internal_error(
                message="Failed to create knowledge", code="KNOWLEDGE_CREATION_FAILED"
            )
