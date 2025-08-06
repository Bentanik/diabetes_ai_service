from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Dict, List, Optional
import logging


logger = logging.getLogger(__name__)


class GeminiLLM:
    _instance: Optional["GeminiLLM"] = None

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.llm: Optional[ChatGoogleGenerativeAI] = None
        self._init_llm()

    def _init_llm(self):
        try:
            self.logger.info("Khởi tạo Gemini LLM...")
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0.2,
                convert_system_message_to_human=True,
            )
            self.logger.info("Khởi tạo Gemini LLM thành công.")
        except Exception as e:
            self.logger.error(f"Lỗi khi khởi tạo Gemini LLM: {e}", exc_info=True)
            raise

    @classmethod
    def get_instance(cls) -> "GeminiLLM":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def invoke(self, messages: List[Dict[str, str]]) -> str:
        if not self.llm:
            raise RuntimeError("LLM chưa được khởi tạo.")
        self.logger.debug(f"Gửi messages: {messages}")
        return self.llm.invoke(messages)
