from app.database import get_collections
from app.database.models import HealthRecordModel
from core.cqrs import CommandHandler, CommandRegistry
from core.result import Result
from ..create_health_record_command import CreateHealthRecordCommand
from shared.messages import UserMessage
from utils import get_logger


@CommandRegistry.register_handler(CreateHealthRecordCommand)
class CreateHealthRecordCommandHandler(CommandHandler):
    def __init__(self):
        super().__init__()
        self.logger = get_logger(__name__)
        self.collections = get_collections()

    async def execute(self, command: CreateHealthRecordCommand) -> Result[None]:
        self.logger.info(f"Tạo dữ liệu sức khỏe người dùng: {command.user_id}")

        try:
            health_record = HealthRecordModel(
                user_id=command.user_id,
                patient_id=command.patient_id,
                type=command.type,
                value=command.value,
                unit=command.unit,
                subtype=command.subtype,
                timestamp=command.timestamp,
            )
            is_update = await self.collections.health_records.insert_one(
                health_record.to_dict()
            )
            if is_update.acknowledged:
                return Result.success(
                    message=UserMessage.HEALTH_RECORD_CREATED.message,
                    code=UserMessage.HEALTH_RECORD_CREATED.code,
                )

            return Result.failure(
                message=UserMessage.HEALTH_RECORD_FAILED.message,
                code=UserMessage.HEALTH_RECORD_FAILED.code,
            )
        except Exception as e:
            self.logger.error(f"Lỗi khi tạo dữ liệu sức khỏe người dùng: {e}", exc_info=True)
            return Result.failure(message=UserMessage.HEALTH_RECORD_FAILED.message, code=UserMessage.HEALTH_RECORD_FAILED.code)