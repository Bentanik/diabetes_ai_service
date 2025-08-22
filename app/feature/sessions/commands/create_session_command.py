from dataclasses import dataclass
from core.cqrs import Command


@dataclass
class CreateSessionCommand(Command):
    """
    Command tạo phiên trò chuyện

    Attributes:
        user_id (str): ID của người dùng
        title (str): Tiêu đề của phiên trò chuyện
    """

    user_id: str
    title: str
