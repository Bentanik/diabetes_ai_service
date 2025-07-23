from typing import Union, Any, Optional, TypeVar
from core.cqrs.base import Command, Query
from core.cqrs.command_registry import CommandRegistry
from core.cqrs.query_registry import QueryRegistry
from core.result import Result
from utils import get_logger

# Type variable để maintain type information
T = TypeVar("T")


class Mediator:
    """Unified mediator for commands and queries"""

    _logger = get_logger("Mediator")

    @classmethod
    async def send(
        cls, request: Union[Command, Query], context: Optional[dict] = None
    ) -> Result[Any]:
        """
        Universal dispatcher for commands and queries

        Args:
            request: Command or Query object
            context: Optional context (user info, transaction, etc.)

        Returns:
            Result object
        """
        try:
            if isinstance(request, Command):
                cls._logger.debug(f"Sending command: {type(request).__name__}")
                return await CommandRegistry.dispatch(request, context)
            elif isinstance(request, Query):
                cls._logger.debug(f"Sending query: {type(request).__name__}")
                return await QueryRegistry.dispatch(request, context)
            else:
                cls._logger.error(f"Unknown request type: {type(request)}")
                return Result.bad_request(  # Thay đổi từ Result.error
                    message=f"Unknown request type: {type(request).__name__}",
                    code="UNKNOWN_REQUEST_TYPE",
                )

        except Exception as e:
            cls._logger.error(
                f"Mediator error: {str(e)}", exc_info=True
            )  # Thêm exc_info
            return Result.internal_error(
                message=f"Mediator error: {str(e)}", code="MEDIATOR_ERROR"
            )

    @classmethod
    def get_registered_handlers(cls) -> dict:
        """Get all registered handlers"""
        return {
            "commands": CommandRegistry.get_registered_handlers(),
            "queries": QueryRegistry.get_registered_handlers(),
        }
