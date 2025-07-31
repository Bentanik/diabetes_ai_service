import asyncio
import json
from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI
from utils import get_logger


class Reflection:
    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm

    def _format_history(self, history: List[dict]) -> str:
        formatted = []
        for msg in history:
            role = msg.get("role", "")
            content = ""
            if "parts" in msg:
                content = " ".join(part["text"] for part in msg["parts"] if "text" in part)
            elif "content" in msg:
                content = msg["content"]
            formatted.append(f"{role.upper()}: {content}")
        return "\n".join(formatted)

    async def __call__(self, chat_history: List[dict], last_n: int = 50) -> str:
        trimmed_history = chat_history[-last_n:] if len(chat_history) > last_n else chat_history
        formatted_history = self._format_history(trimmed_history)

        prompt = (
            "Bạn là trợ lý AI. Dưới đây là lịch sử hội thoại giữa người dùng và hệ thống.\n"
            "Nhiệm vụ của bạn là viết lại câu hỏi cuối cùng của người dùng sao cho có thể hiểu độc lập, "
            "không cần dựa vào bối cảnh trước đó. Nếu không cần viết lại thì giữ nguyên.\n\n"
            f"Lịch sử hội thoại:\n{formatted_history}\n\n"
            "Chỉ trả về câu hỏi viết lại, KHÔNG trả lời câu hỏi."
        )

        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, lambda: self.llm.invoke(prompt))
        print(f"Reflection LLM trả lời:\n{response.content}")
        return response.content.strip()


async def main():
    from core.llm.load_llm import get_gemini_llm
    from rag.retriever import Retriever
    from utils import get_logger

    logger = get_logger(__name__)

    retriever = Retriever()
    llm = get_gemini_llm()
    reflection = Reflection(llm=llm)

    # Lịch sử hội thoại mẫu đúng format (list dict có 'role' và 'content')
    chat_history = [
        {"role": "user", "content": "Xin chào, tôi muốn biết về bệnh tiểu đường"}
    ]

    # Gọi reflection async, last_n=1 lấy câu cuối
    standalone_query = await reflection(chat_history, last_n=1)
    logger.info(f"Câu hỏi viết lại: {standalone_query}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
