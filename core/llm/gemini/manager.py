import asyncio
import logging
from typing import Dict, List, Optional

from cachetools import TTLCache
from .schemas import Message, Role
from .client import GeminiClient
from .config import GeminiConfig


class GeminiChatManager:
    def __init__(self, config: Optional[GeminiConfig] = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.sessions: TTLCache[str, List[Message]] = TTLCache(maxsize=10_000, ttl=600)
        self.user_locks: Dict[str, asyncio.Lock] = {}
        self.prompts: Dict[str, str] = {}
        self.config = config or GeminiConfig()

    async def _get_user_lock(self, user_id: str) -> asyncio.Lock:
        if user_id not in self.user_locks:
            self.user_locks[user_id] = asyncio.Lock()
        return self.user_locks[user_id]

    def set_system_prompt(self, user_id: str, prompt: str):
        self.prompts[user_id] = prompt
        self.logger.info(f"Cập nhật prompt cho user_id={user_id}")

    def get_system_prompt(self, user_id: str) -> str:
        return self.prompts.get(user_id, "Bạn là trợ lý AI thông minh.")

    async def chat(self, user_id: str, user_message: str, history: Optional[List[Message]] = None) -> Message:
        user_lock = await self._get_user_lock(user_id)
        prompt = self.get_system_prompt(user_id)
        async with user_lock:
            if history is not None:
                self.sessions[user_id] = [Message(role=Role.SYSTEM, content=prompt)] + history
                self.logger.info(f"Nhận lịch sử chat từ bên ngoài cho user_id={user_id}")
            elif user_id not in self.sessions:
                self.sessions[user_id] = [Message(role=Role.SYSTEM, content=prompt)]
                self.logger.info(f"Tạo session mới cho user_id={user_id}")
            
            self.sessions[user_id].append(Message(role=Role.USER, content=user_message))
            messages = self.sessions[user_id]

            client = GeminiClient.get_instance(
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                model_name=self.config.model_name,
            )
            response_text = client.invoke(messages)
            message_res = Message(role=Role.ASSISTANT, content=response_text)
            self.sessions[user_id].append(message_res)
            print(f"Phản hồi cho {user_id}: {response_text}")
            print(f"Lịch sử session hiện tại cho {user_id}: {self.sessions[user_id]}")

            return message_res
