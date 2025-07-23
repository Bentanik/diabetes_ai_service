# app/core/base.py
from abc import ABC, abstractmethod
from typing import Any, TypeVar, Generic
from dataclasses import dataclass
from datetime import datetime


# Base Command & Query
class Command(ABC):
    """Base class cho tất cả commands (write operations)"""

    pass


class Query(ABC):
    """Base class cho tất cả queries (read operations)"""

    pass


# Generic result type
TResult = TypeVar("TResult")


# Base Handlers
class CommandHandler(ABC):
    """Base class cho command handlers"""

    @abstractmethod
    async def execute(self, command: Command) -> Any:
        pass


class QueryHandler(Generic[TResult], ABC):
    """Base class cho query handlers"""

    @abstractmethod
    async def handle(self, query: Query) -> TResult:
        pass
