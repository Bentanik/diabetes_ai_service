from typing import Dict, Type, Any, Optional
from core.cqrs.base import Query, QueryHandler
from core.result import Result
from utils import get_logger


class QueryRegistry:
    _handlers: Dict[Type[Query], Type[QueryHandler]] = {}
    _logger = get_logger("QueryRegistry")

    @classmethod
    def register_handler(cls, query_type: Type[Query]):
        """Decorator để tự động đăng ký query handler"""

        def decorator(handler_class: Type[QueryHandler]):
            cls._handlers[query_type] = handler_class
            cls._logger.info(
                f"Đã đăng ký {handler_class.__name__} cho {query_type.__name__}"
            )
            return handler_class

        return decorator

    @classmethod
    async def dispatch(cls, query: Query, context: Optional[dict] = None) -> Any:
        """Tự động điều phối query đến handler tương ứng"""
        query_type = type(query)

        if query_type not in cls._handlers:
            cls._logger.error(f"Không tìm thấy handler cho {query_type.__name__}")
            return Result.failure(
                message=f"Không có handler nào được đăng ký cho {query_type.__name__}",
                code="HANDLER_NOT_FOUND",
            )

        try:
            handler_class = cls._handlers[query_type]
            handler = handler_class()

            if hasattr(handler, "set_context") and context:
                handler.set_context(context)

            cls._logger.info(f"Bắt đầu xử lý {query_type.__name__}")
            result = await handler.execute(query)
            cls._logger.info(f"{query_type.__name__} đã xử lý thành công")

            return result

        except Exception as e:
            cls._logger.error(f"Lỗi khi xử lý {query_type.__name__}: {str(e)}")
            return Result.internal_error(
                message=f"Lỗi khi thực thi query: {str(e)}",
                code="QUERY_EXECUTION_ERROR",
            )

    @classmethod
    def get_registered_handlers(cls) -> Dict[str, str]:
        """Lấy danh sách tất cả các query handler đã được đăng ký"""
        return {
            query.__name__: handler.__name__ for query, handler in cls._handlers.items()
        }

    @classmethod
    def is_registered(cls, query_type: Type[Query]) -> bool:
        """Kiểm tra xem query handler đã được đăng ký hay chưa"""
        return query_type in cls._handlers
