"""Quản lý cấu hình."""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Cấu hình ứng dụng
config = {
    # API Configuration
    "app_title": "AI Service - CarePlan Generator",
    "app_version": "1.0.0",
    "app_description": "AI Service for diabetes care plan generation and measurement analysis",
    # LLM Configuration
    "openrouter_api_key": os.getenv("OPENROUTER_API_KEY"),
    "default_model": "deepseek/deepseek-r1-distill-llama-70b:free",
    "default_temperature": 0.3,
    "openrouter_base_url": "https://openrouter.ai/api/v1",
    # Validation limits
    "max_reason_length": 150,
    "max_feedback_length": 250,
}


# Hàm getter
def get_config(key: str, default=None):
    """Lấy giá trị cấu hình theo key."""
    return config.get(key, default)


def get_api_key():
    """Lấy OpenRouter API key."""
    key = config["openrouter_api_key"]
    if not key:
        raise ValueError("Missing OPENROUTER_API_KEY in environment variables")
    return key
