from typing import Dict, Type, Any, Optional
from core.cqrs.base import Query, QueryHandler
from core.result import Result
from utils import get_logger


class QueryRegistry:
    _handlers: Dict[Type[Query], Type[QueryHandler]] = {}
    _logger = get_logger("QueryRegistry")

    @classmethod
    def register_handler(cls, query_type: Type[Query]):
        """Decorator để auto register query handler"""

        def decorator(handler_class: Type[QueryHandler]):
            cls._handlers[query_type] = handler_class
            cls._logger.info(
                f"Registered {handler_class.__name__} for {query_type.__name__}"
            )
            return handler_class

        return decorator

    @classmethod
    async def dispatch(cls, query: Query, context: Optional[dict] = None) -> Any:
        """Auto dispatch query to appropriate handler"""
        query_type = type(query)

        if query_type not in cls._handlers:
            available_queries = [q.__name__ for q in cls._handlers.keys()]
            cls._logger.error(f"No handler for {query_type.__name__}")
            return Result.error(
                message=f"No handler registered for {query_type.__name__}",
                code="HANDLER_NOT_FOUND",
            )

        try:
            handler_class = cls._handlers[query_type]
            handler = handler_class()

            # Inject context if handler supports it
            if hasattr(handler, "set_context") and context:
                handler.set_context(context)

            cls._logger.info(f"Dispatching {query_type.__name__}")
            result = await handler.handle(query)
            cls._logger.info(f"{query_type.__name__} completed successfully")

            return result

        except Exception as e:
            cls._logger.error(f"Error executing {query_type.__name__}: {str(e)}")
            return Result.internal_error(
                message=f"Error executing query: {str(e)}", code="QUERY_EXECUTION_ERROR"
            )

    @classmethod
    def get_registered_handlers(cls) -> Dict[str, str]:
        """Get all registered query handlers"""
        return {
            query.__name__: handler.__name__ for query, handler in cls._handlers.items()
        }

    @classmethod
    def is_registered(cls, query_type: Type[Query]) -> bool:
        """Check if query handler is registered"""
        return query_type in cls._handlers
