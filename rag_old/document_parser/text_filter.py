import re


class TextFilter:
    """Bộ lọc văn bản - Filter và clean text từ PDF"""

    URL_PATTERN = r"@?https?://[^\s]+|@?www\.|[a-zA-Z0-9.-]+\.(org|com|net|edu|gov|vn)"
    PAGE_NUMBER_PATTERN = r"^\d+/\d+$|^\bPage\s+\d+\b|^\bTrang\s+\d+\b|^\d+\s*$"
    TIMESTAMP_PATTERN = r"\d{2}:\d{2}\s*/"

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

        # Loại bỏ các đoạn chứa URL, metadata, hoặc dấu thời gian
        if remove_urls and re.search(TextFilter.URL_PATTERN, text, re.IGNORECASE):
            return ""
        if remove_page_numbers and (
            re.search(TextFilter.PAGE_NUMBER_PATTERN, text, re.IGNORECASE)
            or re.search(TextFilter.TIMESTAMP_PATTERN, text)
        ):
            return ""

        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Loại bỏ URL và email patterns trong dòng
            if remove_urls:
                url_patterns = [
                    r"@?https?://[^\s]+",  # http://, https://, @https://
                    r"@?www\.[^\s]+",  # www.example.com, @www.example.com
                    r"@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",  # @domain.com
                ]
                for pattern in url_patterns:
                    line = re.sub(pattern, "", line, flags=re.IGNORECASE)

            # Loại bỏ pattern số trang trong dòng
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
        if not text or not text.strip():
            return True

        text_lower = text.strip().lower()

        # Các pattern metadata phổ biến
        metadata_patterns = [
            TextFilter.URL_PATTERN,  # URLs, Wikipedia, tên miền
            TextFilter.PAGE_NUMBER_PATTERN,  # Số trang
            TextFilter.TIMESTAMP_PATTERN,  # Dấu thời gian như "00:25 /"
            r"^[a-z]:\\\\.+",  # Đường dẫn file Windows
            r"^/[a-z/]+",  # Đường dẫn file Unix
            r"^\d+$",  # Chỉ là số
        ]

        for pattern in metadata_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True

        # Kiểm tra các đoạn ngắn chứa dấu gạch chéo hoặc từ khóa metadata
        if len(text) < 50 and ("/" in text_lower or "wikipedia" in text_lower):
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
