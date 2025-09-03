# interactive_test.py
import asyncio
import os
import sys
from datetime import datetime, timedelta
from bson import ObjectId

# Thêm root vào path để import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database.manager import initialize_database
from app.feature.chat.commands.create_chat_command import CreateChatCommand
from core.cqrs import Mediator
from core.result import Result
from utils import get_logger
from app.database import get_collections
from app.database.models import UserProfileModel, HealthRecordModel
from app.database.models import ChatSessionModel  # Cần để tạo/tìm session

logger = get_logger(__name__)


async def seed_user_profile(user_id: str, full_name: str, age: int, gender: str, diabetes_type: str):
    db = get_collections()
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
    now = datetime.utcnow()

    # Đường huyết
    glucose_values = [9.2, 8.7, 9.5] if user_id == "user_001" else [7.8, 8.1, 7.6]
    for i, value in enumerate(glucose_values):
        record = HealthRecordModel(
            user_id=user_id,
            patient_id=patient_id,
            type="Đường huyết",
            value=value,
            unit="mmol/l",
            timestamp=now - timedelta(days=i)
        ).to_dict()
        await db.health_records.insert_one(record)

    # Huyết áp
    bp_sys_values = [145, 142] if user_id == "user_001" else [138, 135]
    for i, sys in enumerate(bp_sys_values):
        record_sys = HealthRecordModel(
            user_id=user_id,
            patient_id=patient_id,
            type="Huyết áp",
            value=sys,
            unit="mmHg",
            subtype="Tâm thu",
            timestamp=now - timedelta(days=i)
        ).to_dict()
        await db.health_records.insert_one(record_sys)

        record_dia = HealthRecordModel(
            user_id=user_id,
            patient_id=patient_id,
            type="Huyết áp",
            value=90 if user_id == "user_001" else 85,
            unit="mmHg",
            subtype="Tâm trương",
            timestamp=now - timedelta(days=i)
        ).to_dict()
        await db.health_records.insert_one(record_dia)

    logger.info(f"✅ Đã tạo chỉ số sức khỏe cho {user_id}")


async def seed_all_data():
    db = get_collections()
    logger.info("🌱 BẮT ĐẦU SEED DỮ LIỆU MẪU")

    USER_IDS = ["user_001", "user_002"]

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
    profile = await db.user_profiles.find_one({"user_id": user_id})
    if not profile:
        return "Không tìm thấy hồ sơ"
    return f"{profile['full_name']} ({profile['age']} tuổi, {profile['diabetes_type']})"


async def show_user_status(user_id: str):
    db = get_collections()
    print("\n" + "👤" * 40)
    profile = await db.user_profiles.find_one({"user_id": user_id})
    if not profile:
        print("❌ Không tìm thấy hồ sơ người dùng")
        return

    print(f"📋 Người dùng: {profile['full_name']}")
    print(f"🆔 ID: {user_id}")
    print(f"🎂 Tuổi: {profile['age']}")
    print(f"🚻 Giới tính: {profile['gender']}")
    print(f"🩺 Loại tiểu đường: {profile['diabetes_type']}")

    # Lấy chỉ số gần nhất
    glu = await db.health_records.find_one(
        {"user_id": user_id, "type": "BloodGlucose"},
        sort=[("timestamp", -1)]
    )
    bp_sys = await db.health_records.find_one(
        {"user_id": user_id, "type": "BloodPressure", "subtype": "tâm thu"},
        sort=[("timestamp", -1)]
    )

    print(f"🩸 Đường huyết gần nhất: {glu['value']:.1f} mmol/l" if glu else "🩸 Đường huyết: Chưa có")
    print(f"💓 Huyết áp tâm thu: {bp_sys['value']:.0f} mmHg" if bp_sys else "💓 Huyết áp: Chưa có")
    print("👤" * 40 + "\n")


async def get_or_create_session(user_id: str, session_id: str = None) -> str:
    db = get_collections()
    if session_id:
        # Kiểm tra xem session có tồn tại và thuộc user không
        session = await db.chat_sessions.find_one({"_id": ObjectId(session_id), "user_id": user_id})
        if session:
            logger.info(f"✅ Tiếp tục sử dụng session: {session_id}")
            return session_id
        else:
            logger.warning(f"Session {session_id} không tồn tại hoặc không thuộc user {user_id}. Tạo mới.")

    # Tạo session mới
    new_session = ChatSessionModel(
        user_id=user_id,
        title="Interactive Test Session"
    )
    result = await db.chat_sessions.insert_one(new_session.to_dict())
    new_session._id = result.inserted_id
    logger.info(f"🆕 Đã tạo session mới: {new_session.id}")
    return str(new_session.id)


async def list_user_sessions(user_id: str):
    db = get_collections()
    cursor = db.chat_sessions.find({"user_id": user_id}).sort("created_at", -1).limit(10)
    sessions = await cursor.to_list(length=10)
    if not sessions:
        print("❌ Không có session nào.")
        return None

    print(f"\n📂 Danh sách session của {user_id}:")
    for i, sess in enumerate(sessions):
        created = sess["created_at"].strftime("%Y-%m-%d %H:%M")
        updated = sess["updated_at"].strftime("%Y-%m-%d %H:%M")
        print(f"  {i+1}. [{sess['_id']}] - {created} → {updated}")
    return [str(s["_id"]) for s in sessions]


async def chat_loop(user_id: str):
    """Vòng lặp chat tương tác – có hỗ trợ session_id"""
    print(f"\n💬 Bắt đầu chat với người dùng: {user_id}")
    print("📌 Nhập:")
    print("   - 'quit': thoát")
    print("   - 'switch': đổi người dùng")
    print("   - 'status': xem thông tin người dùng")
    print("   - 'session new': tạo session mới")
    print("   - 'session list': xem danh sách session")
    print("   - 'clear': dọn màn hình")
    print("-" * 60)

    # Hỏi người dùng muốn dùng session nào
    print("\n🔄 Quản lý session:")
    print("1. Tạo session mới")
    print("2. Chọn từ danh sách session cũ")
    choice = input("👉 Chọn (1/2, Enter để tạo mới): ").strip()

    session_id = None
    if choice == "2":
        session_ids = await list_user_sessions(user_id)
        if session_ids:
            try:
                idx = int(input(f"👉 Chọn số (1-{len(session_ids)}): ")) - 1
                if 0 <= idx < len(session_ids):
                    session_id = session_ids[idx]
                else:
                    print("Số không hợp lệ, tạo session mới.")
            except:
                print("Lỗi nhập liệu, tạo session mới.")
    # Nếu không chọn → tạo mới
    session_id = await get_or_create_session(user_id, session_id)

    print(f"\n🔗 Đang dùng session: {session_id}")
    print("Bạn có thể sao chép ID này để tiếp tục sau.")

    while True:
        try:
            question = input("\n🗨️  Bạn: ").strip()
            if not question:
                continue

            # Xử lý lệnh
            if question.lower() == "quit":
                print("👋 Tạm biệt! Kết thúc phiên chat.")
                break
            elif question.lower() == "switch":
                return
            elif question.lower() == "status":
                await show_user_status(user_id)
                continue
            elif question.lower() == "session list":
                await list_user_sessions(user_id)
                continue
            elif question.lower() == "session new":
                session_id = await get_or_create_session(user_id)
                print(f"🆕 Đã chuyển sang session mới: {session_id}")
                continue
            elif question.lower() == "clear":
                os.system('cls' if os.name == 'nt' else 'clear')
                continue

            print("🧠 AI đang suy nghĩ...")

            command = CreateChatCommand(
                user_id=user_id,
                content=question,
                session_id=session_id  # ← Gửi session_id
            )
            result: Result = await Mediator.send(command)

            if result.is_success:
                response = result.data.content.strip()
                print(f"\n🤖 AI: {response}")
            else:
                print(f"❌ Lỗi: {result.message}")

        except KeyboardInterrupt:
            print("\n👋 Đã thoát bởi người dùng.")
            break
        except Exception as e:
            logger.error(f"Lỗi trong chat loop: {e}")
            print("❌ Có lỗi xảy ra, vui lòng thử lại.")


async def main():
    print("🔧 Khởi tạo cơ sở dữ liệu...")
    await initialize_database()
    print("✅ Khởi tạo thành công\n")

    await seed_all_data()

    USER_IDS = ["user_001", "user_002"]

    print("\n" + "🚀" * 50)
    print("      CHÀO MỪNG BẠN ĐẾN VỚI CHẾ ĐỘ TEST TƯƠNG TÁC")
    print("      Nhập câu hỏi như người dùng thật để kiểm tra AI")
    print("🚀" * 50)

    while True:
        print(f"\n👥 Chọn người dùng để chat:")
        for i, uid in enumerate(USER_IDS, 1):
            info = await get_user_info(uid)
            print(f"  {i}. {uid} → {info}")

        print(f"  {len(USER_IDS) + 1}. Thoát")

        try:
            choice = input(f"\n👉 Nhập số (1-{len(USER_IDS)+1}): ").strip()
            if not choice.isdigit():
                print("❌ Vui lòng nhập số hợp lệ.")
                continue

            choice = int(choice)
            if choice == len(USER_IDS) + 1:
                print("👋 Tạm biệt!")
                break
            elif 1 <= choice <= len(USER_IDS):
                user_id = USER_IDS[choice - 1]
                await show_user_status(user_id)
                await chat_loop(user_id)  # Quay lại chọn nếu gõ 'switch'
            else:
                print("❌ Lựa chọn không hợp lệ.")
        except Exception as e:
            logger.error(f"Lỗi chọn người dùng: {e}")
            print("❌ Có lỗi xảy ra, vui lòng thử lại.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Đã thoát bởi người dùng.")