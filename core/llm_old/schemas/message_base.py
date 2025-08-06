from abc import ABC, abstractmethod


class BaseMessage(ABC):
    @abstractmethod
    def to_dict(self) -> dict:
        pass
