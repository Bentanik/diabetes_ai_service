# simple_rag.py
from core.llm.gemini.client import GeminiClient
from typing import List

def rag_chat(message: str, retrieved_data: List[str], history: List = None) -> str:
    """
    RAG đơn giản: đưa data tìm được vào context
    """
    client = GeminiClient()
    
    # Tạo context từ data retrieved
    if retrieved_data:
        context = "=== DỮ LIỆU LIÊN QUAN ===\n"
        for i, data in enumerate(retrieved_data, 1):
            context += f"{i}. {data}\n"
        context += "========================"
    else:
        context = "Không có dữ liệu liên quan."
    
    system_prompt = """
    Bạn là trợ lý AI. Sử dụng thông tin trong phần DỮ LIỆU LIÊN QUAN để trả lời câu hỏi.
    Nếu không có thông tin phù hợp, hãy nói rõ.
    """
    
    response = client.prompt(
        message=message,
        system_prompt=system_prompt,
        context=context,  # 👈 Context chứa data từ RAG
        history=history or []
    )
    
    return response

# Sử dụng
def main():
    retrieved_data = [
        "Python là ngôn ngữ lập trình bậc cao, dễ học",
        "Python có cú pháp đơn giản với indentation",
        "Python hỗ trợ OOP và functional programming",
        "Python do Nguyễn Mai Viết Vỹ tạo ra"
    ]
    
    message = "Python do ai tạo ra vậy"
    
    response = rag_chat(
        message=message,
        retrieved_data=retrieved_data
    )
    
    print("🤖 Trả lời:")
    print(response)

if __name__ == "__main__":
    main()
