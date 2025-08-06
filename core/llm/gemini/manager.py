import asyncio
import logging
from typing import Dict, List

from cachetools import TTLCache
from .schemas import Message, Role
from .client import GeminiClient


class GeminiChatManager:
    def __init__(self, system_prompt: str = "Bạn là trợ lý AI thông minh."):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.sessions: TTLCache[str, List[Message]] = TTLCache(maxsize=10_000, ttl=600)
        self.user_locks: Dict[str, asyncio.Lock] = {}
        self.system_prompt = system_prompt

    async def _get_user_lock(self, user_id: str) -> asyncio.Lock:
        if user_id not in self.user_locks:
            self.user_locks[user_id] = asyncio.Lock()
        return self.user_locks[user_id]

    async def chat(self, user_id: str, user_message: str) -> str:
        user_lock = await self._get_user_lock(user_id)
        async with user_lock:
            if user_id not in self.sessions:
                self.sessions[user_id] = [
                    Message(role=Role.SYSTEM, content=self.system_prompt)
                ]
                self.logger.info(f"Tạo session mới cho user_id={user_id}")

            self.sessions[user_id].append(Message(role=Role.USER, content=user_message))
            messages = self.sessions[user_id]

            client = GeminiClient.get_instance(max_tokens=512)
            response_text = client.invoke(messages)

            self.sessions[user_id].append(
                Message(role=Role.ASSISTANT, content=response_text)
            )

            self.logger.debug(f"Phản hồi cho user_id={user_id}: {response_text}")
            return response_text
