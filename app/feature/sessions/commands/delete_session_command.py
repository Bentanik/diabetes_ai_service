from dataclasses import dataclass
from core.cqrs import Command


@dataclass
class DeleteSessionCommand(Command):
    """
    Command xóa phiên trò chuyện

    Attributes:
        session_id (str): ID của phiên trò chuyện
    """

    session_id: str
