from typing import Dict, Type, Any, Optional
from core.cqrs.base import Command, CommandHandler
from core.result import Result
from utils import get_logger


class CommandRegistry:
    _handlers: Dict[Type[Command], Type[CommandHandler]] = {}
    _logger = get_logger("CommandRegistry")

    @classmethod
    def register_handler(cls, command_type: Type[Command]):
        """Decorator để auto register command handler"""

        def decorator(handler_class: Type[CommandHandler]):
            cls._handlers[command_type] = handler_class
            cls._logger.info(
                f"Registered {handler_class.__name__} for {command_type.__name__}"
            )
            return handler_class

        return decorator

    @classmethod
    async def dispatch(cls, command: Command, context: Optional[dict] = None) -> Any:
        """Auto dispatch command to appropriate handler"""
        command_type = type(command)

        if command_type not in cls._handlers:
            available_commands = [cmd.__name__ for cmd in cls._handlers.keys()]
            cls._logger.error(f"❌ No handler for {command_type.__name__}")
            return Result.error(
                message=f"No handler registered for {command_type.__name__}",
                code="HANDLER_NOT_FOUND",
            )

        try:
            handler_class = cls._handlers[command_type]
            handler = handler_class()

            # Inject context if handler supports it
            if hasattr(handler, "set_context") and context:
                handler.set_context(context)

            cls._logger.info(f"🚀 Dispatching {command_type.__name__}")
            result = await handler.execute(command)
            cls._logger.info(f"{command_type.__name__} completed successfully")

            return result

        except Exception as e:
            cls._logger.error(f"Error executing {command_type.__name__}: {str(e)}")
            return Result.internal_error(
                message=f"Error executing command: {str(e)}",
                code="COMMAND_EXECUTION_ERROR",
            )

    @classmethod
    def get_registered_handlers(cls) -> Dict[str, str]:
        """Get all registered command handlers"""
        return {
            cmd.__name__: handler.__name__ for cmd, handler in cls._handlers.items()
        }

    @classmethod
    def is_registered(cls, command_type: Type[Command]) -> bool:
        """Check if command handler is registered"""
        return command_type in cls._handlers
