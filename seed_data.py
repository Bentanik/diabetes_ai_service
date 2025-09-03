# seed_data.py
import asyncio
import os
import sys
from datetime import datetime, timedelta

from app.database.db_collections import DBCollections

# Thêm root vào path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import get_collections
from app.database.models import UserProfileModel, HealthRecordModel
from utils import get_logger

logger = get_logger(__name__)
async def seed_user_profile(user_id: str, full_name: str, age: int, gender: str, diabetes_type: str):
    db = get_collections()
    profile = UserProfileModel(
        user_id=user_id,
        patient_id=f"P{user_id[-3:]}",
        full_name=full_name,
        age=age,
        gender=gender,
        bmi=24.5 if gender == "Nam" else 23.8,
        diabetes_type=diabetes_type,
        insulin_schedule="Buổi sáng và buổi tối",
        treatment_method="Insulin + thuốc uống",
        complications=["Bệnh võng mạc"] if user_id == "user_001" else ["Bệnh thần kinh"],
        past_diseases=["Tăng huyết áp"],
        lifestyle="Ăn nhiều rau, ít tinh bột, đi bộ 30 phút mỗi ngày"
    )
    await db.user_profiles.update_one(
        {"user_id": user_id},
        {"$set": profile.to_dict()},
        upsert=True
    )
    logger.info(f"✅ Đã tạo hồ sơ cho {user_id} - {full_name}")

async def seed_health_records(user_id: str, patient_id: str):
    now = datetime.utcnow()
    db = get_collections()

    # Dữ liệu đường huyết (3 ngày gần nhất)
    glucose_values = [9.2, 8.7, 9.5] if user_id == "user_001" else [7.8, 8.1, 7.6]
    for i, value in enumerate(glucose_values):
        record = HealthRecordModel(
            user_id=user_id,
            patient_id=patient_id,
            type="Đường huyết",
            value=value,
            unit="mmol/l",
            timestamp=now - timedelta(days=i)
        )
        await db.health_records.insert_one(record.to_dict())

    # Dữ liệu huyết áp (2 ngày gần nhất)
    bp_sys_values = [145, 142] if user_id == "user_001" else [138, 135]
    for i, sys in enumerate(bp_sys_values):
        # Tâm thu
        record_sys = HealthRecordModel(
            user_id=user_id,
            patient_id=patient_id,
            type="Huyết áp",
            value=sys,
            unit="mmHg",
            subtype="tâm thu",
            timestamp=now - timedelta(days=i)
        )
        await db.health_records.insert_one(record_sys.to_dict())

        # Tâm trương
        dia = 90 if user_id == "user_001" else 85
        record_dia = HealthRecordModel(
            user_id=user_id,
            patient_id=patient_id,
            type="Huyết áp",
            value=dia,
            unit="mmHg",
            subtype="tâm trương",
            timestamp=now - timedelta(days=i)
        )
        await db.health_records.insert_one(record_dia.to_dict())

    logger.info(f"✅ Đã tạo chỉ số sức khỏe cho {user_id}")

async def seed_all():
    logger.info("🌱 BẮT ĐẦU SEED DỮ LIỆU MẪU")
    db = get_collections()
    # Xóa dữ liệu cũ
    await db.user_profiles.delete_many({"user_id": {"$in": ["user_001", "user_002"]}})
    await db.health_records.delete_many({"user_id": {"$in": ["user_001", "user_002"]}})
    logger.info("🧹 Đã xóa dữ liệu cũ")

    # Seed user_001
    await seed_user_profile(
        user_id="user_001",
        full_name="Nguyễn Văn A",
        age=65,
        gender="Nam",
        diabetes_type="Loại 2"
    )
    user1 = await db.user_profiles.find_one({"user_id": "user_001"})
    await seed_health_records("user_001", user1["patient_id"])

    # Seed user_002
    await seed_user_profile(
        user_id="user_002",
        full_name="Trần Thị B",
        age=58,
        gender="Nữ",
        diabetes_type="Loại 2"
    )
    user2 = await db.user_profiles.find_one({"user_id": "user_002"})
    await seed_health_records("user_002", user2["patient_id"])

    logger.info("🎉 SEED DỮ LIỆU HOÀN TẤT!")

async def main():
    from app.database.manager import initialize_database
    await initialize_database()
    await seed_all()

if __name__ == "__main__":
    asyncio.run(main())