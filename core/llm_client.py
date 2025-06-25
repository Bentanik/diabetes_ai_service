import os
from typing import Optional

from dotenv import load_dotenv
from langchain_core.utils.utils import secret_from_env
from langchain_community.chat_models import ChatOpenAI
from pydantic import Field, SecretStr

load_dotenv()


class ChatOpenRouter(ChatOpenAI):
    openai_api_key: Optional[SecretStr] = Field(
        default_factory=lambda: SecretStr(os.getenv("OPENROUTER_API_KEY", "")),
        alias="api_key",
    )

    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"openai_api_key": "OPENROUTER_API_KEY"}

    def __init__(self, openai_api_key: Optional[str] = None, **kwargs):
        key = openai_api_key or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise ValueError(
                "Missing OpenRouter API key. Please set OPENROUTER_API_KEY in your .env file."
            )

        model = kwargs.pop("model", "meta-llama/llama-3.3-8b-instruct:free")

        super().__init__(
            model=model,
            api_key=key,
            base_url="https://openrouter.ai/api/v1",
            **kwargs,
        )


care_plan_llm = ChatOpenRouter(temperature=0.3)  # LLM cho lập lịch đường huyết
note_record_llm = ChatOpenRouter(
    temperature=0.3
)  # LLM cho phân tích ghi chú đo đường huyết


def get_chatbot_llm():
    return ChatOpenRouter(temperature=0.3)  # LLM cho chatbot


async def query_care_plan_llm(prompt: str):
    return await care_plan_llm.ainvoke(prompt)


async def query_note_record_llm(prompt: str):
    return await note_record_llm.ainvoke(prompt)
