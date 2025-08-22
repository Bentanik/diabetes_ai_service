import os
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

class OpenRouterLLM:
    def __init__(
        self,
        model="openai/gpt-oss-20b:free",
        qa_prompt=None,
        external_qa_prompt=None,
        system_prompt=None,
    ):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment")
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        self.model = model

        # Prompt truyền từ bên ngoài vào, nếu không có thì để None
        self.qa_prompt = qa_prompt
        self.external_qa_prompt = external_qa_prompt
        self.system_prompt = system_prompt

    async def generate_async(self, prompt: str, **kwargs) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=kwargs.get("max_tokens", 1500),
            temperature=kwargs.get("temperature", 0.6),
        )
        if not response.choices or len(response.choices) == 0:
            raise RuntimeError("No choices returned from OpenRouter API")
        return response.choices[0].message.content.strip()

    async def chat_async(
        self,
        question: str,
        context: str = "",
        history: list = None,
        system_prompt: str = None,
        prompt_template: str = None,
        **kwargs
    ) -> str:
        messages = []

        # System prompt ưu tiên truyền vào hàm, nếu không thì lấy của object, nếu vẫn không có thì bỏ qua
        sys_prompt = system_prompt or self.system_prompt
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})

        # History
        if history:
            for msg in history:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    messages.append(msg)
                else:
                    raise ValueError(f"Invalid message in history: {msg}")
                    
        prompt = prompt_template or self.qa_prompt
        if not prompt:
            raise ValueError("Prompt template must be provided either via argument or class attribute.")

        # Format prompt
        user_content = prompt.format(context=context, question=question)
        messages.append({"role": "user", "content": user_content})

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", 1500),
            temperature=kwargs.get("temperature", 0.4),
        )
        if not response.choices or len(response.choices) == 0:
            raise RuntimeError("No choices returned from OpenRouter API")
        return response.choices[0].message.content.strip()
