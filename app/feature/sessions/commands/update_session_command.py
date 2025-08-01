from dataclasses import dataclass
from core.cqrs import Command


@dataclass
class UpdateSessionCommand(Command):
    """
    Command cập nhật phiên trò chuyện

    Attributes:
        session_id (str): ID của phiên trò chuyện
        title (str): Tiêu đề của phiên trò chuyện
    """

    session_id: str
    title: str
