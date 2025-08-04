import asyncio
import re
import logging
import json
from typing import List, Optional, Set, Dict
from pathlib import Path

import nltk
from nltk.corpus import stopwords

from rag.types import LanguageInfo

try:
    from underthesea import word_tokenize as vi_tokenize

    HAS_VIETNAMESE_TOKENIZER = True
except ImportError:
    HAS_VIETNAMESE_TOKENIZER = False
    vi_tokenize = None

logger = logging.getLogger(__name__)


class MultilingualTokenizer:
    """Tokenizer hỗ trợ đa ngôn ngữ với JSON stopwords"""

    _english_stopwords: Optional[set] = None
    _vietnamese_stopwords: Optional[set] = None
    _nltk_initialized: bool = False
    _json_stopwords_cache: Dict[str, Set[str]] = {}

    @classmethod
    async def _ensure_nltk_data(cls) -> None:
        """Đảm bảo NLTK data đã được download"""
        if cls._nltk_initialized:
            return

        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            logger.info("Đang tải stopwords NLTK...")
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: nltk.download("stopwords", quiet=True)
            )

        cls._english_stopwords = set(stopwords.words("english"))
        cls._nltk_initialized = True

    @classmethod
    def _load_json_stopwords(cls, language: str) -> Set[str]:
        """Load stopwords từ JSON file"""
        if language in cls._json_stopwords_cache:
            return cls._json_stopwords_cache[language]

        # Tìm file JSON trong data directory
        current_dir = Path(__file__).parent
        json_file = current_dir.parent / "data" / f"{language}_stopwords.json"

        if not json_file.exists():
            logger.warning(f"Không tìm thấy file JSON stopwords: {json_file}")
            return set()

        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            stopwords_set = set(data.get("stopwords", []))
            cls._json_stopwords_cache[language] = stopwords_set
            logger.info(f"Loaded {len(stopwords_set)} stopwords from {json_file}")

            return stopwords_set

        except Exception as e:
            logger.error(f"Lỗi khi tải stopwords JSON cho {language}: {e}")
            return set()

    @classmethod
    async def _get_vietnamese_stopwords(cls) -> Set[str]:
        """Lấy stopwords tiếng Việt"""
        if cls._vietnamese_stopwords is not None:
            return cls._vietnamese_stopwords

        json_stopwords = cls._load_json_stopwords("vietnamese")
        if json_stopwords:
            cls._vietnamese_stopwords = json_stopwords
            return cls._vietnamese_stopwords

        return cls._vietnamese_stopwords

    async def tokenize_text(self, text: str, lang_info: LanguageInfo) -> List[str]:
        """
        Tokenize text theo ngôn ngữ được detect

        Args:
            text: Text cần tokenize
            lang_info: Thông tin ngôn ngữ từ LanguageDetector

        Returns:
            List các token đã loại bỏ stopwords
        """
        await self._ensure_nltk_data()

        # Load stopwords theo ngôn ngữ
        if lang_info["language"] == "vietnamese":
            vietnamese_stopwords = await self._get_vietnamese_stopwords()
        else:
            vietnamese_stopwords = set()

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._tokenize_text_sync, text, lang_info, vietnamese_stopwords
        )

    def _tokenize_text_sync(
        self, text: str, lang_info: LanguageInfo, vietnamese_stopwords: Set[str]
    ) -> List[str]:
        """Synchronous tokenization logic"""
        if lang_info["language"] == "vietnamese" and HAS_VIETNAMESE_TOKENIZER:
            try:
                if vi_tokenize is not None:
                    tokens = vi_tokenize(text)
                else:
                    tokens = text.split()
                return [
                    t
                    for t in tokens
                    if t.lower().strip() not in vietnamese_stopwords and t.strip()
                ]
            except Exception as e:
                logger.warning(
                    f"Vietnamese tokenization failed: {e}, sử dụng tokenization"
                )

        tokens = re.findall(r"\b\w+\b", text.lower())

        if lang_info["language"] == "vietnamese":
            return [t for t in tokens if t not in vietnamese_stopwords]
        else:
            return [t for t in tokens if t not in self._english_stopwords]

    async def remove_stopwords_preserve_format(
        self, text: str, lang_info: LanguageInfo
    ) -> str:
        """
        Remove stopwords nhưng giữ nguyên định dạng văn bản: không chèn dấu cách sai, không gạch dưới, giữ dấu câu đúng chỗ.
        """
        await self._ensure_nltk_data()

        if lang_info["language"] == "vietnamese":
            stopwords_set = await self._get_vietnamese_stopwords()
        else:
            stopwords_set = self._english_stopwords or set()

        if not stopwords_set:
            return text

        # Tokenize
        if (
            lang_info["language"] == "vietnamese"
            and HAS_VIETNAMESE_TOKENIZER
            and vi_tokenize
        ):
            raw_tokens = vi_tokenize(text, format="list")
        else:
            raw_tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

        # Remove stopwords
        filtered_tokens = []
        for token in raw_tokens:
            clean_token = token.lower().strip()
            if clean_token in stopwords_set and re.match(r"^\w+$", token):
                continue
            filtered_tokens.append(token)

        # Smart join tokens
        result = ""
        for i, token in enumerate(filtered_tokens):
            if i == 0:
                result += token
                continue

            prev = filtered_tokens[i - 1]

            # Không thêm space trước dấu câu
            if token in ",.:;!?%)>»”":
                result += token
            # Không thêm space sau dấu mở ngoặc
            elif prev in "($[“«":
                result += token
            # Không thêm space giữa số và % hoặc :
            elif (prev.isdigit() and token in ["%", ":"]) or (
                token.isdigit() and prev in ["(", "$"]
            ):
                result += token
            else:
                result += " " + token

        return result.strip()

    async def tokenize_preserve_format(self, text: str, lang_info: LanguageInfo) -> str:
        """
        Tokenize text nhưng giữ nguyên format (chỉ remove stopwords)
        """
        return await self.remove_stopwords_preserve_format(text, lang_info)

    def count_tokens(self, text: str, lang_info: LanguageInfo) -> int:
        """
        Đếm số token trong text (synchronous version cho performance)

        Args:
            text: Text cần đếm
            lang_info: Thông tin ngôn ngữ

        Returns:
            Số lượng token
        """
        if not text:
            return 0

        # Ước tính nhanh dựa trên ngôn ngữ
        if lang_info["language"] == "vietnamese":
            return len(text) // 3
        elif lang_info["language"] == "english":
            return len(text) // 4
        else:
            return int(len(text) / 3.5)

    @classmethod
    def clear_stopwords_cache(cls) -> None:
        cls._json_stopwords_cache.clear()
        cls._vietnamese_stopwords = None
        cls._english_stopwords = None
        cls._nltk_initialized = False
