import os
from .dto import GeminiConfig
class Config:
    """Cấu hình chung cho hệ thống"""
    # Gemini config
    GEMINI_MODEL_NAME = "gemini-2.0-flash"
    GEMINI_TEMPERATURE = 0.7
    GEMINI_MAX_TOKENS = 2048
    GEMINI_TOP_P = 0.95
    GEMINI_TOP_K = 40
    
    # System prompts
    DEFAULT_SYSTEM_PROMPT = """Bạn là một trợ lý AI thông minh và hữu ích. 
    Hãy trả lời câu hỏi một cách chính xác và chi tiết dựa trên context được cung cấp."""
    
    @classmethod
    def get_gemini_config(cls) -> GeminiConfig:
        return GeminiConfig(
            model_name=cls.GEMINI_MODEL_NAME,
            temperature=cls.GEMINI_TEMPERATURE,
            max_output_tokens=cls.GEMINI_MAX_TOKENS,
            top_p=cls.GEMINI_TOP_P,
            top_k=cls.GEMINI_TOP_K
        )
