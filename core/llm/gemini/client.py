from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from typing import List, Dict, Optional
from dotenv import load_dotenv
import os

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise EnvironmentError("GOOGLE_API_KEY chưa được thiết lập!")

class GeminiClient:
    def __init__(self, model: str = "gemini-2.0-flash", temperature: float = 0.7):
        self.llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)

    def prompt(self,
               message: str,
               system_prompt: Optional[str] = None,
               context: Optional[str] = None,
               history: Optional[List[Dict[str, str]]] = None) -> str:
        
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

    async def prompt_async(self,
                           message: str,
                           system_prompt: Optional[str] = None,
                           context: Optional[str] = None,
                           history: Optional[List[Dict[str, str]]] = None) -> str:
        
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
