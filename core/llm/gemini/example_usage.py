"""
Example sử dụng Gemini với cấu hình động

File này demo cách:
1. Thay đổi cấu hình Gemini từ bên ngoài
2. Load cấu hình từ database
3. Sử dụng cấu hình khác nhau cho từng user
4. Quản lý cache và monitoring memory usage
"""

import asyncio
from typing import Dict, Any
from .manager import GeminiChatManager
from .config import GeminiConfig
from .utils import (
    create_config,
    create_creative_config,
    create_precise_config,
    update_chat_manager_config,
    validate_config,
    print_cache_stats,
    get_cache_health_report,
    get_memory_usage_estimate,
    force_cleanup_cache
)


async def example_basic_usage():
    """Ví dụ sử dụng cơ bản"""
    print("=== Ví dụ sử dụng cơ bản ===")
    
    # Tạo chat manager với config mặc định
    chat_manager = GeminiChatManager()
    
    # Tạo config tùy chỉnh
    custom_config = create_config(
        model_name="gemini-2.0-flash",
        temperature=0.5,
        max_tokens=2048
    )
    
    # Cập nhật config cho user cụ thể
    user_id = "user_123"
    update_chat_manager_config(chat_manager, user_id, custom_config)
    
    print(f"Config cho {user_id}: {chat_manager.get_user_config(user_id)}")
    
    # Chat với config tùy chỉnh
    response = await chat_manager.chat(
        user_id=user_id,
        user_message="Xin chào! Bạn có thể giúp tôi không?"
    )
    print(f"Response: {response.content}")


async def example_multiple_users():
    """Ví dụ sử dụng cho nhiều user với config khác nhau"""
    print("\n=== Ví dụ nhiều user với config khác nhau ===")
    
    chat_manager = GeminiChatManager()
    
    # User 1: Config creative (cao temperature)
    user1_id = "creative_user"
    creative_config = create_creative_config()
    update_chat_manager_config(chat_manager, user1_id, creative_config)
    
    # User 2: Config precise (thấp temperature)
    user2_id = "precise_user"
    precise_config = create_precise_config()
    update_chat_manager_config(chat_manager, user2_id, precise_config)
    
    print(f"Config cho {user1_id}: {chat_manager.get_user_config(user1_id)}")
    print(f"Config cho {user2_id}: {chat_manager.get_user_config(user2_id)}")
    
    # Chat với cả 2 user
    response1 = await chat_manager.chat(
        user_id=user1_id,
        user_message="Hãy kể một câu chuyện sáng tạo về một con mèo"
    )
    
    response2 = await chat_manager.chat(
        user_id=user2_id,
        user_message="Giải thích chính xác về quá trình quang hợp"
    )
    
    print(f"Creative response: {response1.content[:100]}...")
    print(f"Precise response: {response2.content[:100]}...")


async def example_database_integration():
    """Ví dụ tích hợp với database (mock)"""
    print("\n=== Ví dụ tích hợp database ===")
    
    chat_manager = GeminiChatManager()
    user_id = "db_user"
    
    # Mock: Load config từ database
    mock_db_config = {
        "model_name": "gemini-1.5-pro",
        "temperature": 0.7,
        "max_tokens": 3072
    }
    
    # Tạo config từ database data
    db_config = GeminiConfig.from_dict(mock_db_config)
    
    # Validate config trước khi sử dụng
    if validate_config(db_config):
        update_chat_manager_config(chat_manager, user_id, db_config)
        print(f"Đã load config từ database cho {user_id}: {db_config}")
    else:
        print(f"Config từ database không hợp lệ: {db_config}")
    
    # Chat với config từ database
    response = await chat_manager.chat(
        user_id=user_id,
        user_message="Hãy phân tích một vấn đề phức tạp"
    )
    print(f"Response với config từ DB: {response.content[:100]}...")


async def example_dynamic_config_change():
    """Ví dụ thay đổi config động trong runtime"""
    print("\n=== Ví dụ thay đổi config động ===")
    
    chat_manager = GeminiChatManager()
    user_id = "dynamic_user"
    
    # Bắt đầu với config mặc định
    print(f"Config ban đầu: {chat_manager.get_user_config(user_id)}")
    
    # Chat lần 1
    response1 = await chat_manager.chat(
        user_id=user_id,
        user_message="Hãy trả lời ngắn gọn"
    )
    print(f"Response 1: {response1.content[:50]}...")
    
    # Thay đổi config để có response dài hơn
    long_config = create_config(
        model_name="gemini-2.0-flash",
        temperature=0.6,
        max_tokens=4096
    )
    update_chat_manager_config(chat_manager, user_id, long_config)
    print(f"Đã thay đổi config thành: {chat_manager.get_user_config(user_id)}")
    
    # Chat lần 2 với config mới
    response2 = await chat_manager.chat(
        user_id=user_id,
        user_message="Hãy giải thích chi tiết về AI"
    )
    print(f"Response 2: {response2.content[:100]}...")


async def example_cache_management():
    """Ví dụ quản lý cache và monitoring"""
    print("\n=== Ví dụ quản lý cache và monitoring ===")
    
    # Tạo chat manager với TTL ngắn để demo
    chat_manager = GeminiChatManager(client_cache_ttl=60)  # 1 phút
    
    # Tạo nhiều user để demo cache
    users = [f"user_{i}" for i in range(5)]
    
    for user_id in users:
        # Tạo config khác nhau cho mỗi user
        config = create_config(
            temperature=0.1 + (i * 0.2),
            max_tokens=1024 + (i * 512)
        )
        update_chat_manager_config(chat_manager, user_id, config)
        
        # Chat để tạo client cache
        await chat_manager.chat(
            user_id=user_id,
            user_message=f"Xin chào, tôi là {user_id}"
        )
    
    # In thống kê cache
    print("\n--- Thống kê cache sau khi tạo users ---")
    print_cache_stats(chat_manager)
    
    # In báo cáo sức khỏe cache
    print("\n--- Báo cáo sức khỏe cache ---")
    health_report = get_cache_health_report(chat_manager)
    print(f"Overall Health: {'✅ Healthy' if health_report['overall_health'] else '❌ Unhealthy'}")
    print(f"Client Cache: {health_report['client_cache_health']['status']}")
    print(f"Sessions: {health_report['sessions_health']['status']}")
    
    # In ước tính memory usage
    print("\n--- Ước tính memory usage ---")
    memory_usage = get_memory_usage_estimate(chat_manager)
    for key, value in memory_usage.items():
        print(f"{key}: {value}")
    
    # Đợi một chút để demo TTL
    print("\n--- Đợi 70 giây để demo TTL cleanup ---")
    print("(Trong thực tế, cleanup chạy mỗi 5 phút)")
    
    # Force cleanup để demo
    print("\n--- Force cleanup cache ---")
    await force_cleanup_cache(chat_manager)
    
    # In thống kê sau cleanup
    print("\n--- Thống kê cache sau cleanup ---")
    print_cache_stats(chat_manager)


async def example_memory_monitoring():
    """Ví dụ monitoring memory usage liên tục"""
    print("\n=== Ví dụ monitoring memory liên tục ===")
    
    chat_manager = GeminiChatManager()
    
    # Tạo nhiều user để tăng memory usage
    for i in range(10):
        user_id = f"monitor_user_{i}"
        config = create_config(temperature=0.3, max_tokens=1024)
        update_chat_manager_config(chat_manager, user_id, config)
        
        # Chat để tạo cache
        await chat_manager.chat(
            user_id=user_id,
            user_message=f"Test message {i}"
        )
        
        # In thống kê mỗi 3 user
        if (i + 1) % 3 == 0:
            print(f"\n--- Sau {i + 1} users ---")
            print_cache_stats(chat_manager)
            
            memory_usage = get_memory_usage_estimate(chat_manager)
            print(f"Memory: {memory_usage['total_estimated_memory']}")
            
            health = get_cache_health_report(chat_manager)
            print(f"Health: {'✅' if health['overall_health'] else '❌'}")
    
    # Final cleanup
    print("\n--- Final cleanup ---")
    await force_cleanup_cache(chat_manager)
    print_cache_stats(chat_manager)


async def main():
    """Main function để chạy tất cả examples"""
    try:
        await example_basic_usage()
        await example_multiple_users()
        await example_database_integration()
        await example_dynamic_config_change()
        await example_cache_management()
        await example_memory_monitoring()
        
        print("\n=== Tất cả examples đã chạy xong ===")
        
    except Exception as e:
        print(f"Lỗi khi chạy examples: {e}")


if __name__ == "__main__":
    asyncio.run(main())
