"""
Chat Role Type - Enum định nghĩa các vai trò trong cuộc trò chuyện

File này định nghĩa các vai trò trong cuộc trò chuyện, phân biệt giữa người dùng và AI.
"""

from enum import Enum


class ChatRoleType(str, Enum):
    """
    Enum chứa các vai trò trong cuộc trò chuyện.

    Values:
        USER (str): Người dùng
        AI (str): AI
    """

    USER = "user"
    AI = "ai"
