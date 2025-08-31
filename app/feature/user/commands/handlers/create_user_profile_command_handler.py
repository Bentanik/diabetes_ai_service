from app.database import get_collections
from app.database.models import UserProfileModel
from core.cqrs import CommandHandler, CommandRegistry
from core.result import Result
from ..create_user_profile_command import CreateUserProfileCommand
from shared.messages import UserMessage
from utils import get_logger


@CommandRegistry.register_handler(CreateUserProfileCommand)
class CreateUserProfileCommandHandler(CommandHandler):
    def __init__(self):
        super().__init__()
        self.logger = get_logger(__name__)
        self.collections = get_collections()

    async def execute(self, command: CreateUserProfileCommand) -> Result[None]:
        self.logger.info(f"Tạo hồ sơ người dùng: {command.user_id}")

        try:
            user_profile_exists = await self.collections.user_profiles.count_documents(
                {"user_id": command.user_id}
            )
            if user_profile_exists > 0:
                self.logger.info(f"Hồ sơ người dùng đã tồn tại: {command.user_id}")
                return Result.failure(
                    message=UserMessage.DUPLICATE.message,
                    code=UserMessage.DUPLICATE.code,
                )

            user_profile = UserProfileModel(
                user_id=command.user_id,
                patient_id=command.patient_id,
                full_name=command.full_name,
                age=command.age,
                gender=command.gender,
                bmi=command.bmi,
                diabetes_type=command.diabetes_type,
                insulin_schedule=command.insulin_schedule,
                treatment_method=command.treatment_method,
                complications=command.complications,
                past_diseases=command.past_diseases,
                lifestyle=command.lifestyle,
            )
            await self.collections.user_profiles.insert_one(user_profile.to_dict())

            return Result.success(
                message=UserMessage.CREATED.message,
                code=UserMessage.CREATED.code,
            )
        except Exception as e:
            self.logger.error(f"Lỗi khi tạo hồ sơ người dùng: {e}", exc_info=True)
            return Result.failure(message="Lỗi hệ thống", code="error")