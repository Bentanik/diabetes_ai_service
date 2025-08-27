import re
import unicodedata

def clean_text(text: str) -> str:
    if not text:
        return ""
    # Chuẩn hóa Unicode
    text = unicodedata.normalize('NFC', text)
    # Loại bỏ URL
    text = re.sub(r'https?://[^\s]+', '', text)
    # Loại bỏ timestamp: "2:15 PM Wikipedia"
    text = re.sub(r'\d{1,2}:\d{2} [AP]M.*?Wikipedia', '', text)
    # Loại bỏ footnote: [1], [2]
    text = re.sub(r'\[\s*\d+\s*\]', '', text)
    # Loại bỏ ký hiệu trang: /25, /100
    text = re.sub(r'/\d{1,3},?\s*\d{1,2}:\d{2}', '', text)
    # Ngắt từ bị gãy
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    # Nối câu
    text = re.sub(r'([a-z])\n([a-z])', r'\1 \2', text)
    # Dọn khoảng trắng
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_noisy_chunk(text: str) -> bool:
    """Kiểm tra chunk có quá nhiều ký hiệu, ít chữ không"""
    if len(text.split()) < 10:
        return True
    pipe_ratio = text.count("|") / (len(text.split()) + 1)
    if pipe_ratio > 0.3:
        return True
    if "Wikipedia" in text and "://" not in text:
        return True
    return False