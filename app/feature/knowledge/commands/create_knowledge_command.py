from dataclasses import dataclass
from core.cqrs.base import Command


@dataclass
class CreateKnowledgeCommand(Command):
    """
    Command để tạo cơ sở tri thức mới

    Attributes:
        name: Tên cơ sở tri thức
        description: Mô tả chi tiết về cơ sở tri thức
    """

    name: str
    description: str

    def __post_init__(self):
        """
        Thực hiện validation và làm sạch dữ liệu sau khi khởi tạo

        Raises:
            ValueError: Khi tên hoặc mô tả trống hoặc chỉ chứa khoảng trắng
        """
        # Kiểm tra tên cơ sở tri thức
        if not self.name or not self.name.strip():
            raise ValueError("Tên cơ sở tri thức không được để trống")

        # Kiểm tra mô tả cơ sở tri thức
        if not self.description or not self.description.strip():
            raise ValueError("Mô tả không được để trống")

        # Làm sạch dữ liệu bằng cách loại bỏ khoảng trắng thừa
        self.name = self.name.strip()
        self.description = self.description.strip()
