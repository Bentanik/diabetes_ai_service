import asyncio
import logging
import time
from typing import Dict, List, Optional

from cachetools import TTLCache
from .schemas import Message, Role
from .client import GeminiClient
from .config import GeminiConfig, GeminiConfigManager


class GeminiChatManager:
    def __init__(self, config: Optional[GeminiConfig] = None, client_cache_ttl: int = 1800):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.sessions: TTLCache[str, List[Message]] = TTLCache(maxsize=10_000, ttl=600)
        self.user_locks: Dict[str, asyncio.Lock] = {}
        self.prompts: Dict[str, str] = {}
        
        # Sử dụng config manager thay vì config cố định
        self.config_manager = GeminiConfigManager()
        if config:
            self.config_manager.set_default_config(config)
        
        # Cache cho client instances với TTL và giới hạn kích thước
        self._client_cache: TTLCache[str, GeminiClient] = TTLCache(
            maxsize=1000,  # Giới hạn tối đa 1000 clients
            ttl=client_cache_ttl  # TTL mặc định 30 phút
        )
        
        # Cache thời gian sử dụng cuối cùng
        self._client_last_used: Dict[str, float] = {}
        
        # TTL cho client cache (giây)
        self.client_cache_ttl = client_cache_ttl
        
        # Bắt đầu background task để cleanup
        self._cleanup_task = None
        self._start_cleanup_task()

    def _start_cleanup_task(self):
        """Bắt đầu background task để cleanup client cache"""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_client_cache())

    async def _cleanup_client_cache(self):
        """Background task để cleanup client cache định kỳ"""
        while True:
            try:
                await asyncio.sleep(300)  # Chạy mỗi 5 phút
                await self._cleanup_expired_clients()
            except Exception as e:
                self.logger.error(f"Lỗi trong cleanup task: {e}")

    async def _cleanup_expired_clients(self):
        """Xóa các client đã hết hạn"""
        current_time = time.time()
        expired_clients = []
        
        for user_id, last_used in self._client_last_used.items():
            if current_time - last_used > self.client_cache_ttl:
                expired_clients.append(user_id)
        
        for user_id in expired_clients:
            if user_id in self._client_cache:
                del self._client_cache[user_id]
                self.logger.debug(f"Đã xóa expired client cho user_id={user_id}")
            
            if user_id in self._client_last_used:
                del self._client_last_used[user_id]
        
        if expired_clients:
            self.logger.info(f"Đã cleanup {len(expired_clients)} expired clients")

    async def _get_user_lock(self, user_id: str) -> asyncio.Lock:
        if user_id not in self.user_locks:
            self.user_locks[user_id] = asyncio.Lock()
        return self.user_locks[user_id]

    def set_system_prompt(self, user_id: str, prompt: str):
        self.prompts[user_id] = prompt
        self.logger.info(f"Cập nhật prompt cho user_id={user_id}")

    def get_system_prompt(self, user_id: str) -> str:
        return self.prompts.get(user_id, "Bạn là trợ lý AI thông minh.")

    def set_user_config(self, user_id: str, config: GeminiConfig):
        """Cập nhật cấu hình cho một user cụ thể"""
        self.config_manager.set_config(user_id, config)
        # Xóa client cache để tạo mới với config mới
        if user_id in self._client_cache:
            del self._client_cache[user_id]
        self.logger.info(f"Cập nhật cấu hình Gemini cho user_id={user_id}: {config}")

    def get_user_config(self, user_id: str) -> GeminiConfig:
        """Lấy cấu hình cho một user cụ thể"""
        return self.config_manager.get_config(user_id)

    async def load_user_config_from_database(self, user_id: str):
        """Load cấu hình user từ database"""
        config = await self.config_manager.load_config_from_database(user_id)
        if config:
            self.set_user_config(user_id, config)
            self.logger.info(f"Đã load cấu hình từ database cho user_id={user_id}")
        else:
            self.logger.debug(f"Không có cấu hình trong database cho user_id={user_id}, sử dụng default")

    def _get_or_create_client(self, user_id: str) -> GeminiClient:
        """Lấy hoặc tạo client với cấu hình phù hợp"""
        if user_id in self._client_cache:
            # Cập nhật thời gian sử dụng cuối cùng
            self._client_last_used[user_id] = time.time()
            return self._client_cache[user_id]
        
        # Lấy config cho user
        config = self.get_user_config(user_id)
        
        # Tạo client mới với config
        client = GeminiClient.create_instance(
            model_name=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
        
        # Cache client và thời gian sử dụng
        self._client_cache[user_id] = client
        self._client_last_used[user_id] = time.time()
        
        self.logger.debug(f"Tạo client mới cho user_id={user_id} với config: {config}")
        
        return client

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

            # Lấy client với cấu hình phù hợp cho user
            client = self._get_or_create_client(user_id)
            
            response_text = client.invoke(messages)
            message_res = Message(role=Role.ASSISTANT, content=response_text)
            self.sessions[user_id].append(message_res)
            
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"Phản hồi cho {user_id}: {response_text}")
                self.logger.debug(f"Lịch sử session hiện tại cho {user_id}: {self.sessions[user_id]}")

            return message_res

    def clear_user_config(self, user_id: str):
        """Xóa cấu hình riêng của user, quay về dùng default"""
        self.config_manager.remove_config(user_id)
        if user_id in self._client_cache:
            del self._client_cache[user_id]
        if user_id in self._client_last_used:
            del self._client_last_used[user_id]
        self.logger.info(f"Đã xóa cấu hình riêng cho user_id={user_id}")

    def get_all_user_configs(self) -> Dict[str, GeminiConfig]:
        """Lấy tất cả cấu hình user (debug purpose)"""
        return self._client_cache.copy()

    def get_cache_stats(self) -> Dict[str, any]:
        """Lấy thống kê cache để monitor"""
        return {
            "client_cache_size": len(self._client_cache),
            "client_cache_maxsize": self._client_cache.maxsize,
            "sessions_size": len(self.sessions),
            "sessions_maxsize": self.sessions.maxsize,
            "client_last_used_count": len(self._client_last_used)
        }

    async def cleanup_all_caches(self):
        """Xóa tất cả cache (emergency cleanup)"""
        self._client_cache.clear()
        self._client_last_used.clear()
        self.sessions.clear()
        self.logger.info("Đã xóa tất cả cache")

    async def shutdown(self):
        """Shutdown manager và cleanup resources"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        await self.cleanup_all_caches()
        self.logger.info("Đã shutdown GeminiChatManager")
