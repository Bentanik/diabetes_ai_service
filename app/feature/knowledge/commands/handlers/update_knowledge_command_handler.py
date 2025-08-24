"""
Update Knowledge Command Handler - Xử lý cập nhật cơ sở tri thức

File này định nghĩa UpdateKnowledgeCommandHandler. Handler này thực hiện logic cập nhật thông tin cơ sở tri thức.

Chức năng chính:
- Validate ObjectId format
- Hỗ trợ partial update (chỉ cập nhật các field có giá trị)
- Cập nhật timestamp tự động
- Trả về kết quả thành công hoặc lỗi
"""

from datetime import datetime
from bson import ObjectId
from pymongo import ReturnDocument

from app.database import get_collections
from app.feature.knowledge.commands import UpdateKnowledgeCommand
from core.cqrs import CommandRegistry, CommandHandler
from core.result import Result
from shared.messages import KnowledgeMessage
from utils import get_logger


@CommandRegistry.register_handler(UpdateKnowledgeCommand)
class UpdateKnowledgeCommandHandler(CommandHandler):
    """
    Handler để xử lý UpdateKnowledgeCommand
    """
    
    def __init__(self):
        """Khởi tạo handler"""
        super().__init__()
        self.logger = get_logger(__name__)

    async def execute(self, command: UpdateKnowledgeCommand) -> Result[None]:
        """
        Thực hiện cập nhật cơ sở tri thức
        
        Method này thực hiện các bước sau:
        1. Validate ObjectId format
        2. Xây dựng update fields từ command
        3. Cập nhật timestamp
        4. Thực hiện update trong database
        5. Trả về kết quả
        
        Args:
            command (UpdateKnowledgeCommand): Command chứa thông tin cập nhật

        Returns:
            Result[None]: Kết quả thành công hoặc lỗi với message và code tương ứng
        """
        self.logger.info(f"Cập nhật cơ sở tri thức: {command.id}")

        # Validate ObjectId format
        if not ObjectId.is_valid(command.id):
            self.logger.warning(f"ID không hợp lệ: {command.id}")
            return Result.failure(
                message=KnowledgeMessage.NOT_FOUND.message,
                code=KnowledgeMessage.NOT_FOUND.code,
            )

        # Lấy collection để thao tác với database
        collection = get_collections()

        # Xây dựng update fields - chỉ cập nhật các field có giá trị
        update_fields = {}
        if command.name is not None:
            # Kiểm tra tên cơ sở tri thức đã tồn tại chưa
            exists = await collection.knowledges.count_documents({"name": command.name}) > 0
            if exists:
                return Result.failure(
                    message=KnowledgeMessage.NAME_EXISTS.message,
                    code=KnowledgeMessage.NAME_EXISTS.code,
                )
            update_fields["name"] = command.name
        if command.description is not None:
            update_fields["description"] = command.description
        if command.select_training is not None:
            update_fields["select_training"] = command.select_training

        # Kiểm tra nếu không có field nào để cập nhật
        if not update_fields:
            self.logger.info(f"Không có trường nào để cập nhật cho ID: {command.id}")
            return Result.success(
                message=KnowledgeMessage.NO_UPDATE.message,
                code=KnowledgeMessage.NO_UPDATE.code,
                data=None,
            )

        # Thêm timestamp cập nhật
        update_fields["updated_at"] = datetime.now()

        # Thực hiện update và lấy document sau khi update
        updated_doc = await collection.knowledges.find_one_and_update(
            {"_id": ObjectId(command.id)},
            {"$set": update_fields},
            return_document=ReturnDocument.AFTER,
        )

        # Kiểm tra nếu không tìm thấy document để update
        if not updated_doc:
            self.logger.warning(f"Không tìm thấy cơ sở tri thức với id: {command.id}")
            return Result.failure(
                message=KnowledgeMessage.NOT_FOUND.message,
                code=KnowledgeMessage.NOT_FOUND.code,
            )

        self.logger.info(f"Cơ sở tri thức đã được cập nhật: {command.id}")

        # Trả về kết quả thành công
        return Result.success(
            message=KnowledgeMessage.UPDATED.message,
            code=KnowledgeMessage.UPDATED.code,
            data=None,
        )
