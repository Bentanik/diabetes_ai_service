# seed_data.py
import asyncio
from datetime import datetime, timedelta
import sys
import os

from app.database.manager import initialize_database

# Thêm root vào path để import được các module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import get_collections
from app.database.models import UserProfileModel, HealthRecordModel
from bson import ObjectId

async def seed_data():
    # Lấy kết nối DB
    await initialize_database()
    db = get_collections()
    print("✅ Kết nối MongoDB thành công")

    # Xóa dữ liệu cũ (tùy chọn)
    await db.user_profiles.delete_many({})
    await db.health_records.delete_many({})
    print("🧹 Đã xóa dữ liệu cũ trong user_profiles và health_records")

    # === 1. Tạo hồ sơ người dùng mẫu ===
    user_001 = UserProfileModel(
        user_id="user_001",
        patient_id="PT001",
        full_name="Nguyễn Thị Lan",
        age=65,
        gender="Nữ",
        bmi=26.5,
        diabetes_type="Tuýp 2",
        insulin_schedule="Insulin buổi sáng (10 đơn vị)",
        treatment_method="Uống thuốc kết hợp tiêm insulin",
        complications=["Bệnh tim mạch", "Tổn thương thần kinh"],
        past_diseases=["Tăng huyết áp", "Rối loạn mỡ máu"],
        lifestyle="Ít vận động, ăn nhiều tinh bột, ngủ không đủ."
    )

    user_002 = UserProfileModel(
        user_id="user_002",
        patient_id="PT002",
        full_name="Trần Văn Minh",
        age=42,
        gender="Nam",
        bmi=24.0,
        diabetes_type="Tuýp 1",
        insulin_schedule="Insulin trước mỗi bữa ăn (8-10 đơn vị)",
        treatment_method="Tiêm insulin liên tục",
        complications=["Hạ đường huyết thường xuyên"],
        past_diseases=[],
        lifestyle="Vận động thể thao 3 buổi/tuần, ăn uống điều độ."
    )

    # Lưu vào MongoDB
    result1 = await db.user_profiles.insert_one(user_001.to_dict())
    result2 = await db.user_profiles.insert_one(user_002.to_dict())

    print(f"✅ Đã chèn hồ sơ người dùng: {user_001.full_name} (ID: {result1.inserted_id})")
    print(f"✅ Đã chèn hồ sơ người dùng: {user_002.full_name} (ID: {result2.inserted_id})")

    # === 2. Tạo chỉ số sức khỏe mẫu ===
    now = datetime.now()

    records_user_001 = [
        # Đường huyết
        HealthRecordModel(
            user_id="user_001",
            patient_id="PT001",
            type="BloodGlucose",
            value=9.5,
            unit="mmol/l",
            timestamp=now - timedelta(hours=2)
        ),
        HealthRecordModel(
            user_id="user_001",
            patient_id="PT001",
            type="BloodGlucose",
            value=8.7,
            unit="mmol/l",
            timestamp=now - timedelta(days=1)
        ),
        HealthRecordModel(
            user_id="user_001",
            patient_id="PT001",
            type="BloodGlucose",
            value=10.1,
            unit="mmol/l",
            timestamp=now - timedelta(days=2)
        ),
        # Huyết áp
        HealthRecordModel(
            user_id="user_001",
            patient_id="PT001",
            type="BloodPressure",
            value=145,
            unit="mmHg",
            subtype="tâm thu",
            timestamp=now - timedelta(hours=1)
        ),
        HealthRecordModel(
            user_id="user_001",
            patient_id="PT001",
            type="BloodPressure",
            value=90,
            unit="mmHg",
            subtype="tâm trương",
            timestamp=now - timedelta(hours=1)
        ),
    ]

    records_user_002 = [
        # Đường huyết
        HealthRecordModel(
            user_id="user_002",
            patient_id="PT002",
            type="BloodGlucose",
            value=7.2,
            unit="mmol/l",
            timestamp=now - timedelta(hours=1)
        ),
        HealthRecordModel(
            user_id="user_002",
            patient_id="PT002",
            type="BloodGlucose",
            value=6.8,
            unit="mmol/l",
            timestamp=now - timedelta(days=1)
        ),
        HealthRecordModel(
            user_id="user_002",
            patient_id="PT002",
            type="BloodGlucose",
            value=12.0,
            unit="mmol/l",
            timestamp=now - timedelta(days=2)
        ),
    ]

    # Lưu vào MongoDB
    if records_user_001:
        await db.health_records.insert_many([r.to_dict() for r in records_user_001])
        print(f"✅ Đã chèn {len(records_user_001)} chỉ số cho {user_001.full_name}")

    if records_user_002:
        await db.health_records.insert_many([r.to_dict() for r in records_user_002])
        print(f"✅ Đã chèn {len(records_user_002)} chỉ số cho {user_002.full_name}")

    # === 3. In thông tin để kiểm tra ===
    print("\n🌱 Seed data hoàn tất!")
    print("──────────────────────────────────────")
    print("👤 user_001: Nguyễn Thị Lan")
    print("   - 65 tuổi, nữ, tiểu đường tuýp 2")
    print("   - Biến chứng: tim mạch, tổn thương thần kinh")
    print("   - Đường huyết: 9.5, 8.7, 10.1 mmol/l")
    print("   - Huyết áp: 145/90 mmHg")

    print("\n👤 user_002: Trần Văn Minh")
    print("   - 42 tuổi, nam, tiểu đường tuýp 1")
    print("   - Hạ đường huyết thường xuyên")
    print("   - Đường huyết: 7.2, 6.8, 12.0 mmol/l")

if __name__ == "__main__":
    asyncio.run(seed_data())