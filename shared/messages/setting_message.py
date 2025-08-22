from shared.messages.base import BaseResultCode


class SettingMessage(BaseResultCode):
    UPDATED = ("SETTING_UPDATED", "Cài đặt đã được cập nhật thành công")
    FETCHED = ("SETTING_FETCHED", "Cài đặt đã được lấy thành công")
    NOT_FOUND = ("SETTING_NOT_FOUND", "Cài đặt không tồn tại")
