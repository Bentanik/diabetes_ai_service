import os
import logging
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class OpenRouterLLM:
    def __init__(
        self,
        model="qwen/qwen3-30b-a3b:free",
        base_url="https://openrouter.ai/api/v1",
        temperature=0.7
    ):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment")
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        self.model = model
        self.temperature = temperature

    async def generate(
        self,
        prompt: str,
        temperature: float = None,
        max_tokens: int = None,
        top_p: float = None,
        **kwargs
    ) -> str:
        temp = temperature if temperature is not None else self.temperature

        # Xây dựng messages
        messages = [
            {"role": "system", "content": self.system_prompt} if self.system_prompt else None,
            {"role": "user", "content": prompt}
        ]
        messages = [m for m in messages if m]

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temp,
            **{k: v for k, v in kwargs.items() if k in ["top_p", "frequency_penalty", "presence_penalty"]}
        }

        try:
            response = await self.client.chat.completions.create(**payload)
            response.raise_for_status()
            data = response.json()
            if not data.get("choices") or len(data["choices"]) == 0:
                raise RuntimeError("No response from OpenRouter API")
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"OpenRouter API error: {e}")
            return "Xin lỗi, không thể kết nối đến mô hình. Vui lòng thử lại sau."

    async def close(self):
        await self.client.aclose()