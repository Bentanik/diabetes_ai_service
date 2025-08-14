from dataclasses import dataclass
from core.cqrs import Command
from typing import Optional


@dataclass
class UpdateSessionCommand(Command):
    """
    Command cập nhật phiên trò chuyện

    Attributes:
        session_id (str): ID của phiên trò chuyện
        title (str): Tiêu đề của phiên trò chuyện
        external_knowledge (bool): Sử dụng tri thức bên ngoài
    """

    session_id: str
    title: Optional[str]
    external_knowledge: Optional[bool]
