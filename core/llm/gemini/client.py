import logging
from typing import List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from .schemas import Message

from dotenv import load_dotenv
import os

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise EnvironmentError(
        "GOOGLE_API_KEY chưa được thiết lập! Hãy thêm vào file .env hoặc biến môi trường hệ thống."
    )

logger = logging.getLogger(__name__)


class GeminiClient:
    _default_instance: Optional["GeminiClient"] = None

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_name = model_name or "gemini-2.0-flash"
        self.temperature = temperature if temperature is not None else 0.2
        self.max_tokens = max_tokens if max_tokens is not None else 1024
        self.llm: Optional[ChatGoogleGenerativeAI] = None
        self._init_llm()

    def _init_llm(self):
        try:
            self.logger.info(
                f"Khởi tạo Gemini LLM với model={self.model_name}, temperature={self.temperature}, max_tokens={self.max_tokens}"
            )
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                convert_system_message_to_human=True,
            )
            self.logger.info("Khởi tạo Gemini LLM thành công.")
        except Exception as e:
            self.logger.error(f"Lỗi khi khởi tạo Gemini LLM: {e}", exc_info=True)
            raise

    @classmethod
    def get_default_instance(cls) -> "GeminiClient":
        """Lấy instance mặc định với cấu hình cố định"""
        if cls._default_instance is None:
            cls._default_instance = cls()
        return cls._default_instance

    @classmethod
    def create_instance(
        cls,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> "GeminiClient":
        """Tạo instance mới với cấu hình tùy chỉnh"""
        return cls(
            model_name=model_name, 
            temperature=temperature, 
            max_tokens=max_tokens
        )

    @classmethod
    def get_instance(
        cls,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> "GeminiClient":
        """Lấy instance - nếu có tham số thì tạo mới, không thì dùng default"""
        if model_name is not None or temperature is not None or max_tokens is not None:
            # Có tham số mới -> tạo instance mới
            return cls.create_instance(model_name, temperature, max_tokens)
        else:
            # Không có tham số -> dùng default
            return cls.get_default_instance()

    def invoke(self, messages: List[Message]) -> str:
        if not self.llm:
            raise RuntimeError("LLM chưa được khởi tạo.")
        dict_messages = [
            {"role": msg.role.value, "content": msg.content} for msg in messages
        ]
        self.logger.debug(f"Gửi messages tới Gemini: {dict_messages}")
        response = self.llm.invoke(dict_messages)
        self.logger.debug(f"Phản hồi từ Gemini: {response}")
        return response.content
