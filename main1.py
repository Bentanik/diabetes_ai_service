# simple_rag.py
from core.llm.gemini.client import GeminiClient
from typing import List


# def rag_chat(
#     message: str,
#     retrieved_data: List[str],
#     history: List = None,
#     document_name: str = "tài liệu",
# ) -> str:
#     client = GeminiClient()

#     if not retrieved_data:
#         return f"Xin lỗi, tôi không tìm thấy thông tin về '{message}' trong {document_name} hiện có."

#     context = "=== DỮ LIỆU LIÊN QUAN ===\n"
#     for i, data in enumerate(retrieved_data, 1):
#         context += f"{i}. {data}\n"
#     context += "========================"

#     # Thêm relevant history
#     if history:
#         relevant_history = _get_relevant_history(message, history, max_turns=2)
#         if relevant_history:
#             context += "\n\n=== BỐI CẢNH HỘI THOẠI ===\n"
#             for turn in relevant_history:
#                 context += f"User: {turn['user']}\nAI: {turn['ai']}\n"
#             context += "========================"

#     system_prompt = f"""
#     Bạn là trợ lý AI chuyên nghiệp. 
    
#     NHIỆM VỤ:
#     1. Sử dụng DỮ LIỆU LIÊN QUAN làm nguồn thông tin chính
#     2. Tham khảo BỐI CẢNH HỘI THOẠI để hiểu câu hỏi follow-up
#     3. Trả lời chính xác, không bịa đặt
    
#     VÍ DỤ FOLLOW-UP:
#     - "Còn về vấn đề khác thì sao?" → Cần hiểu "vấn đề khác" là gì từ context
#     - "Thế còn cái đó?" → "cái đó" refer đến gì trong lịch sử
#     """

#     response = client.prompt(
#         message=message, system_prompt=system_prompt, context=context
#     )

#     return response


# def _expand_query_with_history(message: str, history: List) -> str:
#     """Mở rộng query dựa trên history để retrieve tốt hơn"""
#     if not history:
#         return message

#     # Lấy 2 turn gần nhất
#     recent_context = []
#     for turn in history[-2:]:
#         recent_context.append(f"User: {turn.get('user', '')}")
#         recent_context.append(f"AI: {turn.get('ai', '')}")

#     expanded = f"{' '.join(recent_context)} {message}"
#     return expanded


# def _get_relevant_history(message: str, history: List, max_turns: int = 2) -> List:
#     """Lấy history có liên quan đến câu hỏi hiện tại"""
#     # Simple approach: lấy những turn gần nhất
#     return history[-max_turns:] if history else []


# # Sử dụng
# def main():
#     # Simulate conversation
#     history = []

#     # Turn 1
#     print("=== Turn 1 ===")
#     retrieved_data1 = ["Python được tạo bởi Guido van Rossum năm 1991"]
#     response1 = rag_chat("Python do ai tạo ra?", retrieved_data1, history)
#     print("🤖:", response1)

#     # Update history
#     history.append({"user": "Python do ai tạo ra?", "ai": response1})

#     # Turn 2 - Follow-up question
#     print("\n=== Turn 2 ===")
#     retrieved_data2 = ["Guido van Rossum sinh năm 1956 tại Hà Lan"]
#     response2 = rag_chat("Ông ấy sinh năm nào?", retrieved_data2, history)
#     print("🤖:", response2)

#     # Update history
#     history.append({"user": "Ông ấy sinh năm nào?", "ai": response2})

#     # Turn 3 - No relevant data
#     print("\n=== Turn 3 ===")
#     response3 = rag_chat("Ông ấy có con không?", [], history)
#     print("🤖:", response3)


# if __name__ == "__main__":
#     main()


if __name__ == "__main__":
    main()
