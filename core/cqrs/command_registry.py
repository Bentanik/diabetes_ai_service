from typing import Dict, Type, Any, Optional
from core.cqrs.base import Command, CommandHandler
from core.result import Result
from utils import get_logger


class CommandRegistry:
    _handlers: Dict[Type[Command], Type[CommandHandler]] = {}
    _logger = get_logger("CommandRegistry")

    @classmethod
    def register_handler(cls, command_type: Type[Command]):
        """Decorator để tự động đăng ký command handler"""

        def decorator(handler_class: Type[CommandHandler]):
            cls._handlers[command_type] = handler_class
            cls._logger.info(
                f"Đã đăng ký {handler_class.__name__} cho {command_type.__name__}"
            )
            return handler_class

        return decorator

    @classmethod
    async def dispatch(cls, command: Command, context: Optional[dict] = None) -> Any:
        """Tự động điều phối command đến handler tương ứng"""
        command_type = type(command)

        if command_type not in cls._handlers:
            cls._logger.error(f"Không tìm thấy handler cho {command_type.__name__}")
            return Result.failure(
                message=f"Không có handler nào được đăng ký cho {command_type.__name__}",
                code="HANDLER_NOT_FOUND",
            )

        try:
            handler_class = cls._handlers[command_type]
            handler = handler_class()

            if hasattr(handler, "set_context") and context:
                handler.set_context(context)

            cls._logger.info(f"Bắt đầu xử lý {command_type.__name__}")
            result = await handler.execute(command)
            cls._logger.info(f"{command_type.__name__} đã xử lý thành công")

            return result

        except Exception as e:
            cls._logger.error(f"Lỗi khi xử lý {command_type.__name__}: {str(e)}")
            return Result.internal_error(
                message=f"Lỗi khi thực thi command: {str(e)}",
                code="COMMAND_EXECUTION_ERROR",
            )

    @classmethod
    def get_registered_handlers(cls) -> Dict[str, str]:
        """Lấy danh sách tất cả các command handler đã được đăng ký"""
        return {
            cmd.__name__: handler.__name__ for cmd, handler in cls._handlers.items()
        }

    @classmethod
    def is_registered(cls, command_type: Type[Command]) -> bool:
        """Kiểm tra xem command handler đã được đăng ký hay chưa"""
        return command_type in cls._handlers
