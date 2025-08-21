from shared.messages.base import BaseResultCode


class DocumentJobMessage(BaseResultCode):
    FETCHED = ("DOCUMENT_JOB_FETCHED", "Tài liệu đã được lấy thành công")
