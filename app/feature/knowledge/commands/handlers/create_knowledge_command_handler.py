"""
Create Knowledge Command Handler - Xử lý tạo cơ sở tri thức mới

File này định nghĩa CreateKnowledgeCommandHandler. Handler này thực hiện logic tạo mới cơ sở tri thức trong database với validation và error handling.

Chức năng chính:
- Kiểm tra tên cơ sở tri thức đã tồn tại chưa
- Tạo KnowledgeModel từ command
- Lưu vào database
- Trả về kết quả thành công hoặc lỗi
"""

from app.database import get_collections
from app.database.models import KnowledgeModel
from core.result import Result
from core.cqrs import CommandRegistry, CommandHandler
from app.feature.knowledge.commands import CreateKnowledgeCommand
from shared.messages import KnowledgeResult
from utils import get_logger


@CommandRegistry.register_handler(CreateKnowledgeCommand)
class CreateKnowledgeCommandHandler(CommandHandler):
    """
    Handler để xử lý CreateKnowledgeCommand
    """
    
    def __init__(self):
        """Khởi tạo handler"""
        super().__init__()
        self.logger = get_logger(__name__)

    async def execute(self, command: CreateKnowledgeCommand) -> Result[None]:
        """
        Thực hiện tạo cơ sở tri thức mới
        
        Method này thực hiện các bước sau:
        1. Kiểm tra tên cơ sở tri thức đã tồn tại chưa
        2. Tạo KnowledgeModel từ command data
        3. Lưu vào database
        4. Trả về kết quả
        
        Args:
            command (CreateKnowledgeCommand): Command chứa thông tin tạo cơ sở tri thức

        Returns:
            Result[None]: Kết quả thành công hoặc lỗi với message và code tương ứng
        """
        self.logger.info(f"Tạo cơ sở tri thức mới: {command.name}")

        # Lấy collection để thao tác với database
        collection = get_collections()
        
        # Kiểm tra tên cơ sở tri thức đã tồn tại chưa
        exists = await collection.knowledges.count_documents({"name": command.name}) > 0
        if exists:
            self.logger.warning(f"Tên cơ sở tri thức đã tồn tại: {command.name}")
            return Result.failure(
                message=KnowledgeResult.NAME_EXISTS.message,
                code=KnowledgeResult.NAME_EXISTS.code,
            )

        # Tạo KnowledgeModel từ command data
        knowledge = KnowledgeModel(name=command.name, description=command.description)

        # Lưu vào database
        await collection.knowledges.insert_one(knowledge.to_dict())

        self.logger.info(f"Cơ sở tri thức đã được tạo thành công: {command.name}")

        # Trả về kết quả thành công
        return Result.success(
            message=KnowledgeResult.CREATED.message,
            code=KnowledgeResult.CREATED.code,
            data=None,
        )
