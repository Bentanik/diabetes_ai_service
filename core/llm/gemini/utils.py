from typing import List, Dict, Any, Optional
from .schemas import Message
from .config import GeminiConfig, GeminiConfigManager
from .manager import GeminiChatManager


def dicts_to_messages(dicts: List[Dict[str, str]]) -> List[Message]:
    return [Message(role=msg["role"], content=msg["content"]) for msg in dicts]


def create_config_from_dict(config_data: Dict[str, Any]) -> GeminiConfig:
    """Tạo GeminiConfig từ dictionary"""
    return GeminiConfig.from_dict(config_data)


def create_config(
    model_name: str = "gemini-2.0-flash",
    temperature: float = 0.2,
    max_tokens: int = 1024
) -> GeminiConfig:
    """Tạo GeminiConfig với các tham số cụ thể"""
    return GeminiConfig(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )


def update_chat_manager_config(
    chat_manager: GeminiChatManager,
    user_id: str,
    config: GeminiConfig
):
    """Cập nhật cấu hình cho một user trong chat manager"""
    chat_manager.set_user_config(user_id, config)


def get_user_config(chat_manager: GeminiChatManager, user_id: str) -> GeminiConfig:
    """Lấy cấu hình hiện tại của một user"""
    return chat_manager.get_user_config(user_id)


async def load_and_apply_user_config(
    chat_manager: GeminiChatManager,
    user_id: str
) -> bool:
    """Load cấu hình từ database và áp dụng cho user"""
    try:
        await chat_manager.load_user_config_from_database(user_id)
        return True
    except Exception as e:
        chat_manager.logger.error(f"Lỗi khi load config cho user {user_id}: {e}")
        return False


def create_creative_config() -> GeminiConfig:
    """Tạo cấu hình cho creative tasks (cao temperature)"""
    return GeminiConfig(
        model_name="gemini-2.0-flash",
        temperature=0.8,
        max_tokens=2048
    )


def create_precise_config() -> GeminiConfig:
    """Tạo cấu hình cho precise tasks (thấp temperature)"""
    return GeminiConfig(
        model_name="gemini-2.0-flash",
        temperature=0.1,
        max_tokens=1024
    )


def create_long_response_config() -> GeminiConfig:
    """Tạo cấu hình cho long responses"""
    return GeminiConfig(
        model_name="gemini-2.0-flash",
        temperature=0.3,
        max_tokens=4096
    )


def validate_config(config: GeminiConfig) -> bool:
    """Validate cấu hình Gemini"""
    try:
        # Kiểm tra temperature
        if not (0.0 <= config.temperature <= 1.0):
            return False
        
        # Kiểm tra max_tokens
        if config.max_tokens <= 0 or config.max_tokens > 8192:
            return False
        
        # Kiểm tra model_name
        valid_models = ["gemini-2.0-flash", "gemini-2.0-flash-exp", "gemini-1.5-flash", "gemini-1.5-pro"]
        if config.model_name not in valid_models:
            return False
        
        return True
    except Exception:
        return False


# Database settings utilities
async def load_settings_from_database(db_collections) -> Optional[Dict[str, Any]]:
    """
    Load settings từ database
    
    Args:
        db_collections: Database collections object
        
    Returns:
        Dict chứa settings hoặc None nếu không tìm thấy
    """
    try:
        setting = await db_collections.settings.find_one({})
        return setting
    except Exception as e:
        print(f"Lỗi khi load settings từ database: {e}")
        return None


def create_llm_config_from_settings(settings: Dict[str, Any]) -> GeminiConfig:
    """
    Tạo GeminiConfig từ settings database
    
    Args:
        settings: Dict chứa settings từ database
        
    Returns:
        GeminiConfig object
    """
    return GeminiConfig(
        model_name=settings.get("llm_model_name", "gemini-2.0-flash"),
        temperature=settings.get("llm_temperature", 0.2),
        max_tokens=settings.get("llm_max_tokens", 1024)
    )

# Cache management utilities
def get_cache_stats(chat_manager: GeminiChatManager) -> Dict[str, Any]:
    """Lấy thống kê cache để monitor"""
    return chat_manager.get_cache_stats()


def print_cache_stats(chat_manager: GeminiChatManager):
    """In thống kê cache ra console"""
    stats = get_cache_stats(chat_manager)
    print("=== Gemini Cache Statistics ===")
    print(f"Client Cache: {stats['client_cache_size']}/{stats['client_cache_maxsize']}")
    print(f"Sessions: {stats['sessions_size']}/{stats['sessions_maxsize']}")
    print(f"Last Used Tracking: {stats['client_last_used_count']}")
    print("===============================")


async def force_cleanup_cache(chat_manager: GeminiChatManager):
    """Force cleanup tất cả cache"""
    await chat_manager.cleanup_all_caches()
    print("Đã force cleanup tất cả cache")


def is_cache_healthy(chat_manager: GeminiChatManager) -> bool:
    """Kiểm tra cache có khỏe mạnh không"""
    stats = get_cache_stats(chat_manager)
    
    # Kiểm tra client cache có quá lớn không
    client_usage_ratio = stats['client_cache_size'] / stats['client_cache_maxsize']
    if client_usage_ratio > 0.8:  # > 80%
        return False
    
    # Kiểm tra sessions có quá lớn không
    sessions_usage_ratio = stats['sessions_size'] / stats['sessions_maxsize']
    if sessions_usage_ratio > 0.9:  # > 90%
        return False
    
    return True


def get_cache_health_report(chat_manager: GeminiChatManager) -> Dict[str, Any]:
    """Lấy báo cáo sức khỏe cache"""
    stats = get_cache_stats(chat_manager)
    
    client_usage_ratio = stats['client_cache_size'] / stats['client_cache_maxsize']
    sessions_usage_ratio = stats['sessions_size'] / stats['sessions_maxsize']
    
    return {
        "overall_health": is_cache_healthy(chat_manager),
        "client_cache_health": {
            "usage_ratio": client_usage_ratio,
            "status": "healthy" if client_usage_ratio < 0.8 else "warning" if client_usage_ratio < 0.9 else "critical"
        },
        "sessions_health": {
            "usage_ratio": sessions_usage_ratio,
            "status": "healthy" if sessions_usage_ratio < 0.9 else "warning" if sessions_usage_ratio < 0.95 else "critical"
        },
        "recommendations": []
    }


def get_memory_usage_estimate(chat_manager: GeminiChatManager) -> Dict[str, str]:
    """Ước tính memory usage của cache (rough estimate)"""
    stats = get_cache_stats(chat_manager)
    
    # Ước tính memory usage (rough calculation)
    # Mỗi GeminiClient: ~1-2MB
    # Mỗi session: ~0.1-0.5MB
    # Mỗi config: ~0.01MB
    
    client_memory = stats['client_cache_size'] * 1.5  # MB
    sessions_memory = stats['sessions_size'] * 0.3    # MB
    total_memory = client_memory + sessions_memory
    
    return {
        "client_cache_memory": f"{client_memory:.1f} MB",
        "sessions_memory": f"{sessions_memory:.1f} MB", 
        "total_estimated_memory": f"{total_memory:.1f} MB",
        "memory_efficiency": "efficient" if total_memory < 100 else "moderate" if total_memory < 500 else "high"
    }
