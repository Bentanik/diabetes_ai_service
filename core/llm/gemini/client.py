from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from typing import List, Dict, Optional
from dotenv import load_dotenv
import os

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise EnvironmentError("GOOGLE_API_KEY chưa được thiết lập!")


class GeminiClient:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GeminiClient, cls).__new__(cls)
        return cls._instance

    def __init__(self, model: str = "gemini-2.0-flash", temperature: float = 0.7):
        if not self._initialized:
            self.model = model
            self.temperature = temperature
            self.llm = ChatGoogleGenerativeAI(
                model=self.model, temperature=self.temperature
            )
            self._initialized = True

    def set_temperature(self, temperature: float):
        """Thay đổi temperature và tạo lại llm instance"""
        self.temperature = temperature
        self.llm = ChatGoogleGenerativeAI(
            model=self.model, temperature=self.temperature
        )

    def get_temperature(self) -> float:
        """Lấy temperature hiện tại"""
        return self.temperature

    def prompt(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        context: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        
        messages = []

        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        if context:
            messages.append(SystemMessage(content=f"CONTEXT:\n{context}"))

        for msg in history or []:
            if msg["role"] == "human":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "ai":
                messages.append(AIMessage(content=msg["content"]))

        messages.append(HumanMessage(content=message))

        return self.llm.invoke(messages).content

    async def prompt_async(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        context: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:

        messages = []

        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        if context:
            messages.append(SystemMessage(content=f"CONTEXT:\n{context}"))

        for msg in history or []:
            if msg["role"] == "human":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "ai":
                messages.append(AIMessage(content=msg["content"]))

        messages.append(HumanMessage(content=message))

        return (await self.llm.ainvoke(messages)).content


# Sử dụng
if __name__ == "__main__":
    # Tạo instance đầu tiên
    client1 = GeminiClient()
    print(f"Temperature: {client1.get_temperature()}")  # 0.7

    # Tạo instance thứ hai - cùng object
    client2 = GeminiClient()
    print(f"Same instance: {client1 is client2}")  # True

    # Thay đổi temperature
    client1.set_temperature(0.3)
    print(f"New temperature: {client2.get_temperature()}")  # 0.3

    # Test chat
    response = client1.prompt("Hello!")
    print(response)
