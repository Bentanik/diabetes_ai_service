import asyncio
from typing import Dict, List

from core.llm.gemini.manager import GeminiChatManager
from core.llm.gemini.schemas import Message, Role
from core.llm.gemini.utils import messages_to_dicts

def save_to_file(history: List[Dict[str, str]], file_path: str):
    import json
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(
            history,
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

async def main():
    manager = GeminiChatManager()
    user_id = "user123"

    manager.set_system_prompt(user_id, "Bạn là trợ lý AI chuyên về y tế.")
    message_user = Message(role=Role.USER, content="Xin chào, tui bị trầm cảm!")
    message = await manager.chat(user_id, message_user.content)
    message_json = messages_to_dicts([message_user, message])
    save_to_file(message_json, "chat_history.json")


if __name__ == "__main__":
    asyncio.run(main())

