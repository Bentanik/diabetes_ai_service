from pydantic import BaseModel, Field
from typing import Optional


class KnowledgeBaseCreateRequest(BaseModel):
    """
    Schema cho request tạo mới một Knowledge Base (Cơ sở tri thức).

    Attributes:
        name (str): Tên của Knowledge Base. Bắt buộc, độ dài từ 1 đến 255 ký tự.
        description (Optional[str]): Mô tả chi tiết về Knowledge Base. Không bắt buộc, tối đa 1000 ký tự.

    Example:
        {
            "name": "Product FAQ",
            "description": "Cơ sở tri thức về các câu hỏi thường gặp sản phẩm"
        }
    """

    name: str = Field(
        ...,
        description="Tên của knowledge base (bắt buộc, 1-255 ký tự)",
        min_length=1,
        max_length=255,
    )
    description: Optional[str] = Field(
        None,
        description="Mô tả về knowledge base (không bắt buộc, tối đa 1000 ký tự)",
        min_length=0,
        max_length=1000,
    )
