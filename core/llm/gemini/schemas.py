from enum import Enum
from dataclasses import dataclass


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    role: Role
    content: str

    def __repr__(self):
        return f"Message(role={self.role.value}, content={self.content})"
