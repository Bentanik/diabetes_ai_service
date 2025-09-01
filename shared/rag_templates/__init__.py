import os
from jinja2 import Environment, FileSystemLoader, select_autoescape

# Đường dẫn đến thư mục template
_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__))

# Khởi tạo Jinja2 Environment
env = Environment(
    loader=FileSystemLoader(_TEMPLATE_DIR),
    autoescape=select_autoescape(default=True, enabled_extensions=('html', 'xml')),
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=False
)

# Pre-load và kiểm tra các template bắt buộc
_REQUIRED_TEMPLATES = [
    "system_prompt.txt",
    "polite_response.j2",
    "trend_response.j2",
    "rag_only.j2",
    "personalized.j2",
    "no_data_response.j2",
    "response_with_rag.j2",
    "no_profile_response.j2"
]

def _validate_templates():
    """Kiểm tra các template cần thiết có tồn tại không"""
    missing = []
    for template_name in _REQUIRED_TEMPLATES:
        path = os.path.join(_TEMPLATE_DIR, template_name)
        if not os.path.exists(path):
            missing.append(template_name)
    if missing:
        raise FileNotFoundError(f"Các template bị thiếu: {', '.join(missing)}")

# Kiểm tra khi import
try:
    _validate_templates()
except Exception as e:
    import logging
    logging.getLogger(__name__).error(f"[Template] Lỗi khởi tạo: {e}")

def render_template(template_name: str, **kwargs) -> str:
    """
    Render một template với dữ liệu đầu vào.

    Args:
        template_name (str): Tên file template (ví dụ: 'response_natural.j2')
        **kwargs: Các biến để truyền vào template

    Returns:
        str: Nội dung đã được render

    Example:
        render_template("response_natural.j2", question="Đái tháo đường là gì?", contexts=[...])
    """
    try:
        template = env.get_template(template_name)
        return template.render(**kwargs).strip()
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Không thể render template '{template_name}': {e}")
        raise

# Export để dễ import
__all__ = ["render_template", "env"]