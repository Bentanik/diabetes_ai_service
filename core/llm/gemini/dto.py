from dataclasses import dataclass


@dataclass
class GeminiConfig:
    """Cấu hình cho Gemini model"""
    model_name: str = "gemini-2.0-flash"
    temperature: float = 0.7
    max_output_tokens: int = 2048
    top_p: float = 0.95
    top_k: int = 40