from abc import ABC, abstractmethod
from typing import Any, TypeVar, Generic


class Command(ABC):
    pass


class Query(ABC):
    pass


TResult = TypeVar("TResult")


class CommandHandler(ABC):
    @abstractmethod
    async def execute(self, command: Command) -> Any:
        pass


class QueryHandler(Generic[TResult], ABC):
    @abstractmethod
    async def execute(self, query: Query) -> TResult:
        pass
