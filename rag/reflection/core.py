from typing import List
import logging

logger = logging.getLogger(__name__)


class Reflection:
    def __init__(self, llm):
        self.llm = llm

    def _format_history(self, data) -> str:
        lines = []
        for entry in data:
            role = entry.get("role", "").strip()
            content = ""

            if "parts" in entry and isinstance(entry["parts"], list):
                content = " ".join(part.get("text", "") for part in entry["parts"] if isinstance(part, dict))
            elif "content" in entry:
                content = entry["content"]
            elif "text" in entry:
                content = entry["text"]

            if role and content.strip():
                lines.append(f"{role.capitalize()}: {content.strip()}")
        return "\n".join(lines)

    async def __call__(self, chat_history: List[dict], last_items: int = 10) -> str:
        if not chat_history:
            return ""

        recent = chat_history[-last_items:]
        history_str = self._format_history(recent)

        prompt = f"""
Bạn là trợ lý thông minh. Hãy viết lại câu hỏi cuối cùng của người dùng thành dạng tự đứng được (standalone), 
không cần lịch sử để hiểu.

Lịch sử trò chuyện:
{history_str}

---
Yêu cầu:
- Chỉ viết lại câu hỏi cuối (role=user).
- Giải quyết tham chiếu: "nó", "vậy", "ở trên".
- Trả về **chỉ mỗi câu hỏi**, không giải thích.
- Dùng tiếng Việt.

Câu hỏi đã viết lại:"""

        try:
            response = self.llm.generate_content([{"role": "user", "content": prompt}])
            rewritten = response.text.strip()
            logger.info(f"[Reflection] Input: {chat_history[-1]['content']} → Output: {rewritten}")
            return rewritten
        except Exception as e:
            logger.warning(f"[Reflection] Failed: {e}")
            # Fallback
            for msg in reversed(recent):
                if msg.get("role") == "user":
                    return msg.get("content") or ""
            return ""