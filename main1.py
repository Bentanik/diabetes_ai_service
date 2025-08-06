import asyncio

from core.llm.gemini.manager import GeminiChatManager


async def main():
    manager = GeminiChatManager(
        "Bạn là một chuyên gia về bệnh đái tháo đường, không được nhắc đến gì liên quan kiểu tui là AI, bạn phải trả lời bằng tiếng Việt, tui muốn bạn phải trả lời thân thiện và giải thích ra những khái niệm và các vấn đề liên quan đến bệnh đái tháo đường dễ hiểu"
    )
    user_id = "test_user"
    user_message = "Với căn bệnh tiểu đường loại 2 thì tui nên làm gì ổn"

    response = await manager.chat(user_id, user_message)
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
