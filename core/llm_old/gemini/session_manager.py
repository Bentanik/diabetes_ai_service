import asyncio
import logging
from typing import List, Dict, Optional
from .schemas import Message, Role
from .llm import GeminiLLM


class GeminiChatSessionManager:
    def __init__(self, system_prompt: Optional[str] = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.sessions: Dict[str, List[Message]] = {}
        self._lock = asyncio.Lock()

        prompt_text = system_prompt or "Bạn là trợ lý AI thông minh."
        self._system_prompt = Message(Role.SYSTEM, prompt_text)
        self.logger.info(f"System prompt set to: {prompt_text}")

    async def get_history(self, user_id: str) -> List[Message]:
        async with self._lock:
            return self.sessions.get(user_id, [])

    async def append_message(self, user_id: str, role: Role, content: str) -> None:
        async with self._lock:
            if user_id not in self.sessions:
                self.sessions[user_id] = [self._system_prompt]
                self.logger.info(
                    f"Tạo session mới cho user_id={user_id} với system prompt."
                )
            self.sessions[user_id].append(Message(role, content))
            self.logger.debug(
                f"User {user_id} thêm message role={role.value}: {content}"
            )

    async def clear_history(self, user_id: str) -> None:
        async with self._lock:
            if user_id in self.sessions:
                del self.sessions[user_id]
                self.logger.info(f"Xoá lịch sử chat user_id={user_id}")

    async def chat(self, user_id: str, user_message: str) -> str:
        await self.append_message(user_id, Role.USER, user_message)
        async with self._lock:
            messages = self.sessions.get(user_id, [])
        llm = await GeminiLLM.get_instance()
        response = await llm.invoke(messages)
        await self.append_message(user_id, Role.ASSISTANT, response)
        self.logger.debug(f"Phản hồi từ LLM cho user_id={user_id}: {response}")
        return response
