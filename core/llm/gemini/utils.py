from typing import List, Dict
from .schemas import Message


def messages_to_dicts(messages: List[Message]) -> List[Dict[str, str]]:
    return [{"role": msg.role.value, "content": msg.content} for msg in messages]


def dicts_to_messages(dicts: List[Dict[str, str]]) -> List[Message]:
    return [Message(role=msg["role"], content=msg["content"]) for msg in dicts]
