"""
Chat Role Type Enum - Enum cho vai trò của người dùng trong DTO

File này định nghĩa ChatRoleType enum cho DTO layer.
"""

from enum import Enum


class ChatRoleType(str, Enum):
    """
    Enum định nghĩa các vai trò của người dùng trong DTO

    Values:
        USER: Người dùng
        AI: AI
    """

    USER = "user"
    AI = "ai"
