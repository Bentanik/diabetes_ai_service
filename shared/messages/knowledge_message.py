from shared.messages.base import BaseResultCode


class KnowledgeMessage(BaseResultCode):
    CREATED = ("KNOWLEDGE_CREATED", "Cơ sở tri thức đã được tạo thành công")
    NAME_EXISTS = ("KNOWLEDGE_NAME_EXISTS", "Tên cơ sở tri thức đã tồn tại")
    FETCHED = ("KNOWLEDGE_FETCHED", "Cơ sở tri thức đã được lấy thành công")
    NOT_FOUND = ("KNOWLEDGE_NOT_FOUND", "Cơ sở tri thức không tồn tại")
    NO_UPDATE = ("KNOWLEDGE_NO_UPDATE", "Không có thay đổi")
    UPDATED = ("KNOWLEDGE_UPDATED", "Cơ sở tri thức đã được cập nhật thành công")
    DELETED = ("KNOWLEDGE_DELETED", "Cơ sở tri thức đã được xóa thành công")
