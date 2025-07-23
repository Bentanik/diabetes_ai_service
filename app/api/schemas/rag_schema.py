from typing import Optional
from pydantic import BaseModel


class UpdateKnowledgeRequest(BaseModel):
    """
    Request để cập nhật cơ sở tri thức

    Attributes:
        name: Tên cơ sở tri thức
        description: Mô tả cơ sở tri thức
        select_training: Đánh dấu có chọn để huấn luyện hay không
    """

    name: Optional[str] = None
    description: Optional[str] = None
    select_training: Optional[bool] = None
