from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from typing import List, Dict, Optional
from dotenv import load_dotenv
import os

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise EnvironmentError("GOOGLE_API_KEY chưa được thiết lập!")

SYS_PROMPT_GENERAL_KNOWLEDGE = """You are an **AI Knowledge Partner** - a brilliant and passionate guide to understanding. Your mission is not just to answer questions, but to **empower users with deep comprehension**.

## CORE PRINCIPLES:
- **Clarity First**: Break complex ideas into digestible parts using analogies and examples
- **Beyond Surface Level**: Explain the "why" behind facts, connect to broader concepts
- **Natural Conversation**: Write as you would speak - warm, engaging, intelligent

## EXECUTION RULES:

**1. LANGUAGE**: Always respond in **VIETNAMESE** regardless of the question language

**2. PROACTIVE & THOUGHTFUL**: 
- Answer the main question first
- Add valuable context: *"Điều thú vị là..."*, *"Bạn có biết rằng..."*
- Anticipate follow-up questions when relevant

**3. HANDLE AMBIGUITY**: 
If unclear, ask for clarification: *"Để trả lời chính xác, bạn muốn tôi tập trung vào khía cạnh [A] hay [B]?"*

**4. HONESTY & SAFETY**:
- If lacking reliable info: *"Tôi xin lỗi, nhưng tôi không có đủ thông tin đáng tin cậy về chủ đề này."*
- **Refuse** dangerous, illegal, or harmful content requests

**5. STRUCTURE**:
- **Direct summary** answering the core question first
- **Detailed explanation** with concrete examples
- Use **Markdown** (headings, **bold**, lists) for readability

Be the trusted knowledge companion who makes learning exciting!"""



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

    async def prompt_async(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        context: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:

        messages = []

        # 1. Thêm System Prompt (chỉ dẫn hệ thống) - Phần này giữ nguyên, đã rất tốt
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))

        # 2. Thêm lịch sử hội thoại - Phần này giữ nguyên, đã rất tốt
        for msg in history or []:
            if msg["role"] == "human":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "ai":
                messages.append(AIMessage(content=msg["content"]))

        # 3. Xây dựng HumanMessage cuối cùng một cách thông minh (Đây là phần thay đổi chính)
        # Nếu có context, chúng ta sẽ dùng một template để kết hợp context và câu hỏi
        if context:
            # Template này ra lệnh rõ ràng cho LLM phải làm gì với context
            prompt_template = f"""Dựa vào ngữ cảnh dưới đây, hãy trả lời câu hỏi của người dùng theo các bước sau:

        1.  **Trích xuất:** Đầu tiên, hãy đọc kỹ câu hỏi và xác định tất cả các thông tin, sự kiện, và dữ liệu có liên quan đến câu hỏi từ trong 'NGỮ CẢNH'.
        2.  **Tổng hợp:** Tiếp theo, hãy sắp xếp các thông tin đã trích xuất thành một bài giải thích mạch lạc và toàn diện.
        3.  **Trả lời:** Cuối cùng, hãy trình bày bài giải thích đã tổng hợp làm câu trả lời. **Nếu không tìm thấy thông tin để trả lời câu hỏi trong 'NGỮ CẢNH', hãy trả lời một cách lịch sự rằng "Tôi không tìm thấy thông tin liên quan trong tài liệu được cung cấp."**

        --- NGỮ CẢNH ---
        {context}
        --- KẾT THÚC NGỮ CẢNH ---

        Câu hỏi: {message}"""
            messages.append(HumanMessage(content=prompt_template))
        else:
            # Nếu không có context, chỉ cần thêm câu hỏi của người dùng
            messages.append(HumanMessage(content=message))

        # 4. Gọi LLM và trả về kết quả - Phần này giữ nguyên
        return (await self.llm.ainvoke(messages)).content
    
    async def prompt_no_rag_async(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Hàm MỚI, chuyên dùng cho các câu hỏi không cần context (tri thức chung).
        """
        messages = []

        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))

        for msg in history or []:
            if msg["role"] == "human":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "ai":
                messages.append(AIMessage(content=msg["content"]))
        
        messages.append(HumanMessage(content=message))

        return (await self.llm.ainvoke(messages)).content



# Sử dụng
if __name__ == "__main__":
    import asyncio
    async def main():
        client = GeminiClient()
        client.set_temperature(0.2)

        print("--- Test 1: Gọi hàm RAG (prompt_async) với context ---")
        my_context = "Trong một khu rừng nọ, con cáo nổi tiếng với bộ lông màu cam và rất gian xảo. Con thỏ thì có bộ lông trắng và chạy rất nhanh."
        question_1 = "Con cáo có đặc điểm gì?"
        # Hàm này được thiết kế để dùng với context
        response_1 = await client.prompt_async(message=question_1, context=my_context)
        print(f"Question: {question_1}\nAnswer: {response_1}\n")

        print("--- Test 2: Gọi hàm KHÔNG RAG (prompt_no_rag_async) cho tri thức chung ---")
        question_2 = "Thủ đô của nước Pháp là gì?"
        # Dùng hàm mới, và có thể truyền vào một system prompt mạnh mẽ
        response_2 = await client.prompt_no_rag_async(
            message=question_2,
            system_prompt=SYS_PROMPT_GENERAL_KNOWLEDGE
        )
        print(f"Question: {question_2}\nAnswer: {response_2}\n")

    asyncio.run(main())
