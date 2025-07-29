"""
Text Filter Module - Bộ lọc và làm sạch văn bản từ PDF

Module này cung cấp các utility để lọc và làm sạch text được trích xuất từ PDF:
- Loại bỏ URL, email, và các link không cần thiết
- Loại bỏ số trang và metadata (6/15, Page 1, Trang 1, etc.)
- Loại bỏ dòng quá ngắn có thể là noise
- Detect và filter metadata/reference không cần thiết
- Thống kê quá trình làm sạch văn bản
"""

import re


class TextFilter:
    """Bộ lọc văn bản - Filter và clean text từ PDF"""

    @staticmethod
    def clean_text(
        text: str,
        remove_urls: bool = True,
        remove_page_numbers: bool = True,
        remove_short_lines: bool = True,
        min_line_length: int = 3,
    ) -> str:
        """Làm sạch và lọc nội dung văn bản"""
        if not text or not text.strip():
            return ""

        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Loại bỏ URL và email patterns
            if remove_urls:
                url_patterns = [
                    r"@?https?://[^\s]+",  # http://, https://, @https://
                    r"@?www\.[^\s]+",  # www.example.com, @www.example.com
                    r"@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",  # @domain.com
                ]
                for pattern in url_patterns:
                    line = re.sub(pattern, "", line, flags=re.IGNORECASE)

            # Loại bỏ pattern số trang
            if remove_page_numbers:
                page_patterns = [
                    r"\b\d+/\d+\b",  # "6/15", "1/10"
                    r"\bPage\s+\d+\b",  # "Page 1", "Page 10"
                    r"\bTrang\s+\d+\b",  # "Trang 1", "Trang 10"
                    r"\b\d+\s*$",  # Số ở cuối dòng
                ]
                for pattern in page_patterns:
                    line = re.sub(pattern, "", line, flags=re.IGNORECASE)

            # Dọn dẹp khoảng trắng thừa
            line = re.sub(r"\s+", " ", line).strip()

            # Lọc dòng quá ngắn
            if remove_short_lines and len(line) < min_line_length:
                continue

            if line:
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    @staticmethod
    def is_likely_metadata(text: str) -> bool:
        """Kiểm tra xem văn bản có phải là metadata cần được lọc bỏ không"""
        text = text.strip().lower()

        # Các pattern metadata phổ biến
        metadata_patterns = [
            r"^\d+/\d+$",  # Số trang như "6/15"
            r"^page\s+\d+",  # "Page 1"
            r"^trang\s+\d+",  # "Trang 1"
            r"^@?https?://",  # URLs
            r"^@?www\.",  # WWW links
            r"^\d+$",  # Chỉ là số
            r"^[a-z]:\\\\.+",  # Đường dẫn file Windows
            r"^/[a-z/]+",  # Đường dẫn file Unix
        ]

        for pattern in metadata_patterns:
            if re.match(pattern, text):
                return True

        return False

    @staticmethod
    def get_text_statistics(original_text: str, cleaned_text: str) -> dict:
        """Thống kê về quá trình làm sạch văn bản"""
        return {
            "original_length": len(original_text),
            "cleaned_length": len(cleaned_text),
            "original_lines": len(original_text.split("\n")),
            "cleaned_lines": len(cleaned_text.split("\n")),
            "reduction_percentage": round(
                (1 - len(cleaned_text) / max(len(original_text), 1)) * 100, 2
            ),
            "lines_removed": len(original_text.split("\n"))
            - len(cleaned_text.split("\n")),
        }
