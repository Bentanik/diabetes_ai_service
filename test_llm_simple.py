#!/usr/bin/env python3
"""Script test đơn giản cho LLM configuration mới."""

import os
import asyncio
import sys

# Thêm src vào path
sys.path.append("src")

from core.llm_client import get_llm


async def test_current_config():
    """Test cấu hình LLM hiện tại."""
    print("🚀 KIỂM TRA LLM CONFIGURATION")
    print("=" * 50)

    try:
        # Lấy client và thông tin
        client = get_llm()
        info = client.get_provider_info()

        print(f"✅ Base URL: {info['base_url']}")
        print(f"✅ Model: {info['model']}")
        print(f"✅ Temperature: {info['temperature']}")
        print(f"✅ Max Tokens: {info['max_tokens']}")
        print(f"✅ Has API Key: {'Có' if info['has_api_key'] else 'Không'}")

        # Test generate
        print("\n🔄 KIỂM TRA GENERATE...")
        test_prompt = "Chào bạn! Viết 1 câu ngắn bằng tiếng Việt."
        print(f"Prompt: {test_prompt}")

        response = await client.generate(test_prompt)
        print(f"Response: {response[:100]}{'...' if len(response) > 100 else ''}")

        print("\n✅ LLM hoạt động tốt!")

    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return False

    return True


def show_config_examples():
    """Hiển thị ví dụ cấu hình."""
    print("\n📝 VÍ DỤ CẤU HÌNH:")
    print("=" * 50)

    examples = {
        "OpenRouter": {
            "LLM_BASE_URL": "https://openrouter.ai/api/v1",
            "LLM_API_KEY": "sk-or-v1-xxx...",
            "LLM_MODEL": "deepseek/deepseek-r1-distill-llama-70b:free",
        },
        "Localhost (vLLM)": {
            "LLM_BASE_URL": "http://localhost:8000/v1",
            "LLM_API_KEY": "",
            "LLM_MODEL": "meta-llama/Llama-3.3-8B-Instruct",
        },
        "Ollama": {
            "LLM_BASE_URL": "http://localhost:11434/v1",
            "LLM_API_KEY": "",
            "LLM_MODEL": "llama3.2",
        },
        "OpenAI": {
            "LLM_BASE_URL": "https://api.openai.com/v1",
            "LLM_API_KEY": "sk-xxx...",
            "LLM_MODEL": "gpt-3.5-turbo",
        },
    }

    for name, config in examples.items():
        print(f"\n{name}:")
        for key, value in config.items():
            print(f"  {key}={value}")


async def main():
    """Main function."""
    # Test config hiện tại
    success = await test_current_config()

    # Hiển thị examples
    show_config_examples()

    print("\n💡 CÁCH SỬ DỤNG:")
    print("=" * 50)
    print("1. Thiết lập 3 biến môi trường: LLM_BASE_URL, LLM_API_KEY, LLM_MODEL")
    print("2. Restart ứng dụng")
    print("3. Test với script này hoặc gọi API /system/llm-info")

    if not success:
        print("\n⚠️  Lưu ý: Cần cấu hình biến môi trường trước khi test!")


if __name__ == "__main__":
    asyncio.run(main())
