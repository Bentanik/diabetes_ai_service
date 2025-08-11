from dataclasses import dataclass
from typing import Optional, Dict, Any
import logging


@dataclass
class GeminiConfig:
    model_name: str = "gemini-2.0-flash"
    temperature: float = 0.2
    max_tokens: int = 1024
    
    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi sang dictionary"""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeminiConfig":
        """Tạo config từ dictionary"""
        return cls(
            model_name=data.get("model_name", "gemini-2.0-flash"),
            temperature=data.get("temperature", 0.2),
            max_tokens=data.get("max_tokens", 1024)
        )


class GeminiConfigManager:
    """Quản lý cấu hình Gemini động từ database hoặc external source"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._default_config = GeminiConfig()
        self._config_cache: Dict[str, GeminiConfig] = {}
    
    def get_default_config(self) -> GeminiConfig:
        """Lấy cấu hình mặc định"""
        return self._default_config
    
    def set_default_config(self, config: GeminiConfig):
        """Cập nhật cấu hình mặc định"""
        self._default_config = config
        self.logger.info(f"Cập nhật cấu hình mặc định: {config}")
    
    def get_config(self, config_key: str = "default") -> GeminiConfig:
        """Lấy cấu hình theo key (có thể là user_id, session_id, etc.)"""
        if config_key in self._config_cache:
            return self._config_cache[config_key]
        return self._default_config
    
    def set_config(self, config_key: str, config: GeminiConfig):
        """Cập nhật cấu hình cho một key cụ thể"""
        self._config_cache[config_key] = config
        self.logger.info(f"Cập nhật cấu hình cho {config_key}: {config}")
    
    def remove_config(self, config_key: str):
        """Xóa cấu hình cho một key cụ thể"""
        if config_key in self._config_cache:
            del self._config_cache[config_key]
            self.logger.info(f"Xóa cấu hình cho {config_key}")
    
    def clear_cache(self):
        """Xóa tất cả cache cấu hình"""
        self._config_cache.clear()
        self.logger.info("Đã xóa tất cả cache cấu hình")
    
    async def load_config_from_database(self, config_key: str) -> Optional[GeminiConfig]:
        """Load cấu hình từ database (implement theo nhu cầu)"""
        try:
            # TODO: Implement loading từ database
            # Ví dụ:
            # config_data = await database.get_gemini_config(config_key)
            # if config_data:
            #     return GeminiConfig.from_dict(config_data)
            self.logger.debug(f"Chưa implement load config từ database cho {config_key}")
            return None
        except Exception as e:
            self.logger.error(f"Lỗi khi load config từ database: {e}")
            return None
    
    async def save_config_to_database(self, config_key: str, config: GeminiConfig) -> bool:
        """Lưu cấu hình vào database (implement theo nhu cầu)"""
        try:
            # TODO: Implement saving vào database
            # Ví dụ:
            # await database.save_gemini_config(config_key, config.to_dict())
            self.logger.debug(f"Chưa implement save config vào database cho {config_key}")
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi save config vào database: {e}")
            return False
