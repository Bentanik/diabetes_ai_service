from enum import Enum
from dataclasses import dataclass


class Role(Enum):
    """
    Enum định nghĩa vai trò của từng message trong cuộc trò chuyện.
    - SYSTEM: Thông điệp hệ thống, thiết lập ngữ cảnh.
    - USER: Tin nhắn từ người dùng.
    - ASSISTANT: Phản hồi từ AI trợ lý.
    """

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """
    Đại diện một message trong hội thoại.
    Bao gồm vai trò (role) và nội dung (content).
    """

    role: Role
    content: str

    def to_dict(self) -> dict:
        """
        Chuyển đối tượng Message thành dict theo định dạng
        mà API LLM yêu cầu: {'role': str, 'content': str}.
        """
        return {
            "role": self.role.value,
            "content": self.content,
        }
