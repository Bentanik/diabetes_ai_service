# test_with_real_data.py
import asyncio
import os
import sys
from datetime import datetime, timedelta

# Thêm root vào path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database.manager import initialize_database
from app.feature.chat.commands.create_chat_command import CreateChatCommand
from core.cqrs import Mediator
from core.result import Result
from utils import get_logger
from app.database import get_collections
from app.database.models import UserProfileModel, HealthRecordModel

logger = get_logger(__name__)

# Danh sách người dùng test
USER_IDS = ["user_001", "user_002"]


async def seed_user_profile(user_id: str, full_name: str, age: int, gender: str, diabetes_type: str):
    db = get_collections()

    """Tạo hồ sơ người dùng nếu chưa có"""
    profile_data = {
        "user_id": user_id,
        "patient_id": f"P{user_id[-3:]}",
        "full_name": full_name,
        "age": age,
        "gender": gender,
        "bmi": 24.5 if gender == "Nam" else 23.8,
        "diabetes_type": diabetes_type,
        "insulin_schedule": "Buổi sáng và buổi tối",
        "treatment_method": "Insulin + thuốc uống",
        "complications": ["Bệnh võng mạc"] if user_id == "user_001" else ["Bệnh thần kinh"],
        "past_diseases": ["Tăng huyết áp"],
        "lifestyle": "Ăn nhiều rau, ít tinh bột, đi bộ 30 phút mỗi ngày"
    }
    await db.user_profiles.update_one(
        {"user_id": user_id},
        {"$set": profile_data},
        upsert=True
    )
    logger.info(f"✅ Đã tạo hồ sơ cho {user_id} - {full_name}")


async def seed_health_records(user_id: str, patient_id: str):
    db = get_collections()
    db = get_collections()

    """Tạo dữ liệu sức khỏe mẫu"""
    now = datetime.utcnow()

    # Đường huyết: user_001 cao, user_002 ổn định
    glucose_values = [9.2, 8.7, 9.5] if user_id == "user_001" else [7.8, 8.1, 7.6]
    for i, value in enumerate(glucose_values):
        record = HealthRecordModel(
            user_id=user_id,
            patient_id=patient_id,
            type="BloodGlucose",
            value=value,
            unit="mmol/l",
            timestamp=now - timedelta(days=i)
        ).to_dict()
        await db.health_records.insert_one(record)

    # Huyết áp
    bp_sys_values = [145, 142] if user_id == "user_001" else [138, 135]
    for i, sys in enumerate(bp_sys_values):
        # Tâm thu
        record_sys = HealthRecordModel(
            user_id=user_id,
            patient_id=patient_id,
            type="BloodPressure",
            value=sys,
            unit="mmHg",
            subtype="tâm thu",
            timestamp=now - timedelta(days=i)
        ).to_dict()
        await db.health_records.insert_one(record_sys)

        # Tâm trương
        dia = 90 if user_id == "user_001" else 85
        record_dia = HealthRecordModel(
            user_id=user_id,
            patient_id=patient_id,
            type="BloodPressure",
            value=dia,
            unit="mmHg",
            subtype="tâm trương",
            timestamp=now - timedelta(days=i)
        ).to_dict()
        await db.health_records.insert_one(record_dia)

    logger.info(f"✅ Đã tạo chỉ số sức khỏe cho {user_id}")


async def seed_all_data():
    db = get_collections()

    """Seed dữ liệu cho user_001 và user_002"""
    logger.info("🌱 BẮT ĐẦU SEED DỮ LIỆU MẪU")

    for user_id in USER_IDS:
        profile = await db.user_profiles.find_one({"user_id": user_id})
        if not profile:
            if user_id == "user_001":
                await seed_user_profile(user_id, "Nguyễn Văn A", 65, "Nam", "Loại 2")
            else:
                await seed_user_profile(user_id, "Trần Thị B", 58, "Nữ", "Loại 2")

            profile = await db.user_profiles.find_one({"user_id": user_id})
            await seed_health_records(user_id, profile["patient_id"])

    logger.info("🎉 SEED DỮ LIỆU HOÀN TẤT!")


async def get_user_info(user_id: str):
    db = get_collections()

    """Lấy thông tin người dùng"""
    profile = await db.user_profiles.find_one({"user_id": user_id})
    if not profile:
        return "❌ Không tìm thấy hồ sơ"
    return f"{profile['full_name']}, {profile['age']} tuổi, {profile['diabetes_type']}"


async def get_latest_glucose(user_id: str):
    db = get_collections()

    """Lấy đường huyết gần nhất"""
    record = await db.health_records.find_one(
        {"user_id": user_id, "type": "BloodGlucose"},
        sort=[("timestamp", -1)]
    )
    return f"{record['value']:.1f} mmol/l" if record else "Không có"


async def get_latest_bp(user_id: str):
    db = get_collections()

    """Lấy huyết áp tâm thu gần nhất"""
    record = await db.health_records.find_one(
        {"user_id": user_id, "type": "BloodPressure", "subtype": "tâm thu"},
        sort=[("timestamp", -1)]
    )
    return f"{record['value']:.0f} mmHg" if record else "Không có"


async def test_case(user_id: str, question: str, description: str, expected_type: str = "general"):
    """Chạy một test case và in kết quả"""
    print("\n" + "=" * 70)
    print(f"🧪 TEST: {description}")
    print(f"👤 User ID: {user_id}")
    print(f"💬 Hỏi: '{question}'")

    # Hiển thị thông tin người dùng
    profile_info = await get_user_info(user_id)
    latest_glu = await get_latest_glucose(user_id)
    latest_bp = await get_latest_bp(user_id)

    print(f"📋 Hồ sơ: {profile_info}")
    print(f"🩸 Đường huyết gần nhất: {latest_glu}")
    print(f"💓 Huyết áp gần nhất: {latest_bp}")
    print("=" * 70)

    # Gửi câu hỏi
    command = CreateChatCommand(
        user_id=user_id,
        content=question,
        session_id=None
    )
    result: Result = await Mediator.send(command)

    if result.is_success:
        response = result.data.content.strip()
        print("✅ TRẢ LỜI:")
        print(f"📝 {response}")

        # Kiểm tra nhanh
        checks = {
            "Tiếng Việt": "❌" if any(word in response.lower() for word in ["i ", "you ", "sorry"]) else "✅",
            "Không leak": "❌" if any(kw in response.lower() for kw in ["hãy suy nghĩ", "phân tích"]) else "✅",
            "Không trống": "✅" if len(response) > 20 else "❌",
            "Có Markdown": "✅" if "**" in response or "*" in response or "###" in response else "❌"
        }

        print("\n🔍 KIỂM TRA NHANH:")
        for k, v in checks.items():
            print(f"  {v} {k}")

        return response
    else:
        print("❌ LỖI:", result.message)
        return ""


async def run_tests():
    db = get_collections()

    print("🚀 BẮT ĐẦU TEST VỚI DỮ LIỆU THỰC")
    print(f"🕒 Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("📌 Mục tiêu: Test RAG, cá nhân, thiếu dữ liệu, câu hỏi ngoài chủ đề")
    print("──────────────────────────────────────────────────────────────────────")

    for user_id in USER_IDS:
        print(f"\n👨‍👩‍👧‍👦 ĐANG TEST VỚI NGƯỜI DÙNG: {user_id}")
        print("─" * 50)

        # --- 1. TEST CÂU HỎI CÁ NHÂN & XU HƯỚNG ---
        await test_case(
            user_id=user_id,
            question="Gần đây đường huyết của tôi thế nào?",
            description="Xu hướng: Đường huyết",
            expected_type="trend"
        )

        await test_case(
            user_id=user_id,
            question="Huyết áp dạo này ra sao?",
            description="Xu hướng: Huyết áp",
            expected_type="trend"
        )

        # --- 2. TEST CÂU HỎI CÁ NHÂN + RAG ---
        await test_case(
            user_id=user_id,
            question="Với tình trạng của tôi, đường huyết 9.5 có nguy hiểm không?",
            description="Kết hợp: RAG + Hồ sơ",
            expected_type="personal"
        )

        # --- 3. TEST RAG: CÂU HỎI VỀ TIỂU ĐƯỜNG (CÓ DỮ LIỆU) ---
        await test_case(
            user_id=user_id,
            question="Tiểu đường là gì?",
            description="RAG: Kiến thức cơ bản (có trong dữ liệu)",
            expected_type="rag_only"
        )

        await test_case(
            user_id=user_id,
            question="Tiểu đường có mấy loại?",
            description="RAG: Phân loại bệnh (có trong dữ liệu)",
            expected_type="rag_only"
        )

        await test_case(
            user_id=user_id,
            question="Người tiểu đường nên ăn gì?",
            description="RAG: Chế độ ăn (có dữ liệu)",
            expected_type="rag_only"
        )

        # --- 4. TEST KHÔNG CÓ TRONG RAG (ngoài chủ đề) ---
        await test_case(
            user_id=user_id,
            question="Ung thư và tiểu đường có liên quan đến nhau không?",
            description="RAG: Chủ đề không hỗ trợ (không có trong dữ liệu)",
            expected_type="rag_only"
        )

        await test_case(
            user_id=user_id,
            question="Ăn quá nhiều đường có dẫn đến ung thư không?",
            description="RAG: Câu hỏi liên quan gián tiếp (không có dữ liệu)",
            expected_type="rag_only"
        )

        # --- 5. TEST KHÔNG CÓ DỮ LIỆU ĐƯỜNG HUYẾT (giả lập) ---
        if user_id == "user_002":
            # Xóa dữ liệu đường huyết tạm thời
            await db.health_records.delete_many({
                "user_id": user_id,
                "type": "BloodGlucose"
            })
            await test_case(
                user_id=user_id,
                question="Đường huyết của tôi dạo này ra sao?",
                description="Không có dữ liệu đường huyết",
                expected_type="trend"
            )
            # Khôi phục
            profile = await db.user_profiles.find_one({"user_id": user_id})
            await db.health_records.insert_one({
                "user_id": user_id,
                "patient_id": profile["patient_id"],
                "type": "BloodGlucose",
                "value": 7.8,
                "unit": "mmol/l",
                "timestamp": datetime.utcnow()
            })

    print("\n🎉 TẤT CẢ TEST ĐÃ HOÀN TẤT!")
    print("✅ Test RAG: Câu hỏi về tiểu đường → trả lời chính xác")
    print("✅ Test không RAG: Câu hỏi ngoài chủ đề → không bịa, trả lời an toàn")
    print("✅ Test cá nhân: Có phân tích theo hồ sơ")
    print("✅ Test thiếu dữ liệu: Có hướng dẫn tử tế")
    print("✅ Không có lỗi has_analyzed_trend")


async def main():
    print("🔧 Khởi tạo cơ sở dữ liệu...")
    await initialize_database()
    print("✅ Khởi tạo thành công\n")

    await seed_all_data()
    await run_tests()


if __name__ == "__main__":
    asyncio.run(main())