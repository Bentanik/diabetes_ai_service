from typing import Any, Optional
from core.cqrs.command_registry import CommandRegistry
from core.cqrs.query_registry import QueryRegistry
from core.cqrs.base import Command, Query


class Mediator:
    @staticmethod
    async def send(request: Any, context: Optional[dict] = None) -> Any:
        """
        Tự động phân loại và điều phối request đến handler tương ứng

        Parameters:
            request (Any): Một instance của Command hoặc Query.
            context (Optional[dict]): Thông tin ngữ cảnh (context) kèm theo (nếu có)

        Returns:
            Any: Kết quả trả về từ handler (thường là một đối tượng Result).

        Raises:
            TypeError: Nếu đối tượng không phải là Command hoặc Query.
        """
        if isinstance(request, Command):
            return await CommandRegistry.dispatch(request, context)
        elif isinstance(request, Query):
            return await QueryRegistry.dispatch(request, context)
        else:
            raise TypeError("Đối tượng gửi phải là Command hoặc Query")
