import logging
import httpx

logger = logging.getLogger(__name__)


class QwenLLM:
    def __init__(
        self,
        model: str = "qwen2.5:3b-instruct",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.client = httpx.AsyncClient(timeout=60.0)

    async def generate(
        self,
        prompt: str,
        temperature: float = None
    ) -> str:
        temp = temperature if temperature is not None else self.temperature

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temp
            }
        }
        try:
            response = await self.client.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
            return response.json()["response"].strip()
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return "Xin lỗi, mô hình đang bận, vui lòng thử lại sau."

    async def close(self):
        await self.client.aclose()