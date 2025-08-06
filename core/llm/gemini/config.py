from dataclasses import dataclass


@dataclass
class GeminiConfig:
    model_name: str = "gemini-2.0-flash"
    temperature: float = 0.2
    max_tokens: int = 1024
