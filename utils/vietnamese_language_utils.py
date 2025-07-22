import re
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class VietnameseLanguageUtils:
    def __init__(self, vi_words_json: Path = None):
        self._vietnamese_words_cache = None
        self._vietnamese_abbreviations_cache = None
        self._vi_words_json = vi_words_json

    def merge_lines(self, text: str) -> str:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        merged = ""
        for i, line in enumerate(lines):
            if merged and not merged.rstrip().endswith(
                (".", "!", "?", ":", ";", '"', "”", "’")
            ):
                merged += " " + line
            else:
                if merged:
                    merged += "\n"
                merged += line
        return merged

    def clean_vietnamese_text(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(
            r"[^\w\s\u00C0-\u024F\u1E00-\u1EFF.,!?;:()\[\]\"'\n\-%/]", " ", text
        )
        text = re.sub(
            r"([.!?])\s*([A-ZÁÀẮÂÉÈÊÔƠÚÙƯĐÍÌ])",
            r"\1 \2",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r" +", " ", text).strip()
        text = self.merge_lines(text)
        return text

    def _contains_vietnamese(self, text: str) -> bool:
        vietnamese_chars = (
            r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđĐ]"
        )
        return bool(re.search(vietnamese_chars, text, re.IGNORECASE))

    def _get_vietnamese_words(self) -> set:
        if self._vietnamese_words_cache is not None:
            return self._vietnamese_words_cache
        if self._vi_words_json and self._vi_words_json.exists():
            try:
                with open(self._vi_words_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                all_words = set()
                for category, words in data.items():
                    if category != "metadata" and isinstance(words, list):
                        all_words.update(words)
                logger.info(
                    f"Đã load {len(all_words)} từ tiếng Việt từ {self._vi_words_json.name}"
                )
                self._vietnamese_words_cache = all_words
                return all_words
            except Exception as e:
                logger.error(f"Lỗi load file từ JSON: {e}")
        fallback_words = {
            "tôi",
            "bạn",
            "anh",
            "chị",
            "em",
            "mình",
            "ta",
            "họ",
            "ai",
            "gì",
            "là",
            "có",
            "được",
            "làm",
            "đi",
            "đến",
            "về",
            "xin",
            "chào",
            "tốt",
            "đẹp",
            "lớn",
            "nhỏ",
            "mới",
            "cũ",
            "và",
            "với",
            "từ",
            "để",
            "trong",
            "ngoài",
            "ạ",
            "ơi",
            "nhé",
            "rồi",
            "đã",
            "chưa",
            "lý",
            "viên",
            "công",
            "ty",
            "trợ",
            "giúp",
        }
        self._vietnamese_words_cache = fallback_words
        return fallback_words

    def _get_vietnamese_abbreviations(self) -> set:
        if self._vietnamese_abbreviations_cache is not None:
            return self._vietnamese_abbreviations_cache
        if self._vi_words_json and self._vi_words_json.exists():
            try:
                with open(self._vi_words_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                abbreviations = data.get("abbreviations", [])
                abbreviations_set = set(abbreviations)
                self._vietnamese_abbreviations_cache = abbreviations_set
                return abbreviations_set
            except Exception as e:
                logger.error(f"Lỗi load file từ JSON: {e}")
        fallback_abbr = {"ai", "it", "hr", "ceo"}
        self._vietnamese_abbreviations_cache = fallback_abbr
        return fallback_abbr

    def _calculate_vietnamese_score(self, text: str) -> float:
        if not text:
            return 0.0
        words = re.findall(r"\w+", text.lower())
        vietnamese_words_set = self._get_vietnamese_words()
        if not words:
            return 0.0
        vietnamese_abbreviations = self._get_vietnamese_abbreviations()
        vietnamese_word_count = 0
        for word in words:
            if word in vietnamese_words_set:
                vietnamese_word_count += 1
            elif word in vietnamese_abbreviations and len(words) > 2:
                vietnamese_word_count += 0.5
        word_score = vietnamese_word_count / len(words)
        vietnamese_chars = (
            r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđĐ]"
        )
        vietnamese_char_count = len(re.findall(vietnamese_chars, text, re.IGNORECASE))
        total_chars = len(
            re.findall(
                r"[a-zA-ZàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđĐ]",
                text,
            )
        )
        char_score = vietnamese_char_count / total_chars if total_chars > 0 else 0.0
        combined_score = (word_score * 0.7) + (char_score * 0.3)
        return min(1.0, combined_score)

    def detect_language(self, text: str) -> str:
        if not text or not text.strip():
            return "unknown"
        vn_score = self._calculate_vietnamese_score(text)
        contains_vn = self._contains_vietnamese(text)
        ascii_chars = len(re.findall(r"[a-zA-Z]", text))
        total_chars = len(
            re.findall(
                r"[a-zA-ZàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđĐ]",
                text,
            )
        )
        if total_chars == 0:
            return "unknown"
        ascii_ratio = ascii_chars / total_chars
        english_words = len(
            re.findall(
                r"\b(speak|english|hello|world|the|and|is|are|this|that|with|for)\b",
                text.lower(),
            )
        )
        total_words = len(text.split())
        english_word_ratio = english_words / total_words if total_words > 0 else 0
        if contains_vn and english_word_ratio > 0.2:
            return "mixed"
        elif contains_vn and vn_score > 0.1:
            return "vi"
        elif ascii_ratio > 0.9 and not contains_vn:
            return "en"
        else:
            return "unknown"
