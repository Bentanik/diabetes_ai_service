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


async def _calculate_diabetes_score(text_blocks: List[str]) -> float:
    """
    Tính điểm diabetes - weighted average toàn bộ document
    """
    from utils.diabetes_scorer_utils import async_analyze_diabetes_content

    try:
        if not text_blocks:
            return 0.0

        # Clean blocks nhưng giữ nguyên structure
        valid_blocks = []
        for block in text_blocks:
            cleaned = block.strip()
            if cleaned:  # Chỉ giữ blocks có content
                valid_blocks.append(cleaned)
            # Skip empty blocks hoàn toàn

        if not valid_blocks:
            return 0.0

        # Process in batches để tránh overwhelm
        batch_size = 50
        all_analyses = []

        for i in range(0, len(valid_blocks), batch_size):
            batch = valid_blocks[i : i + batch_size]
            tasks = [async_analyze_diabetes_content(block) for block in batch]
            batch_analyses = await asyncio.gather(*tasks, return_exceptions=True)
            all_analyses.extend(batch_analyses)

        # Weighted average calculation
        total_score = 0.0
        total_weight = 0.0
        processed_blocks = 0
        error_blocks = 0

        for analysis, block in zip(all_analyses, valid_blocks):
            # Calculate weight (có thể dùng word count hoặc char count)
            weight = len(block.split())  # hoặc len(block) cho char count
            total_weight += weight

            if isinstance(analysis, Exception):
                error_blocks += 1
            else:
                processed_blocks += 1
                total_score += analysis.final_score * weight
                print(
                    f"Score={analysis.final_score:.3f}, weight={weight}: {block[:50]}..."
                )

        # Handle edge case: all blocks failed
        if total_weight == 0:
            return 0.0

        diabetes_score = total_score / total_weight

        return diabetes_score

    except Exception as e:
        return 0.0


async def main():
    # manager = GeminiChatManager()
    # user_id = "user123"

    # manager.set_system_prompt(user_id, "Bạn là trợ lý AI chuyên về y tế.")
    # message_user = Message(role=Role.USER, content="Xin chào, tui bị trầm cảm!")
    # message = await manager.chat(user_id, message_user.content)
    # message_json = messages_to_dicts([message_user, message])
    # save_to_file(message_json, "chat_history.json")
    hehe = await _calculate_diabetes_score(
        [
            "Bệnh tiểu đường type 2 là bệnh mãn tính. Bệnh nhân có triệu chứng khát nước và đi tiểu nhiều.",
            "Diabetes mellitus affects glucose metabolism. Treatment includes insulin therapy.",
            "Hôm nay trời đẹp, tôi đi chơi công viên với bạn bè.",
            "The patient presented with elevated HbA1c levels.",
            "Biến chứng tiểu đường có thể ảnh hưởng nghiêm trọng đến sức khỏe.",
        ]
    )
    print(hehe)


if __name__ == "__main__":
    asyncio.run(main())
