from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Dict, List, Optional
import logging


logger = logging.getLogger(__name__)


class GeminiLLM:
    _instance: Optional["GeminiLLM"] = None

    def __init__(self, model_name: str = "gemini-2.0-flash", temperature: float = 0.2):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_name = model_name
        self.temperature = temperature
        self.llm: Optional[ChatGoogleGenerativeAI] = None
        self._init_llm()

    def _init_llm(self) -> None:
        try:
            self.logger.info(f"Khởi tạo Gemini LLM với model '{self.model_name}'...")
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature,
                convert_system_message_to_human=True,
            )
            self.logger.info("Khởi tạo Gemini LLM thành công.")
        except Exception as e:
            self.logger.error(f"Lỗi khi khởi tạo Gemini LLM: {e}", exc_info=True)
            raise

    @classmethod
    def get_instance(
        cls,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.2,
        force_reload: bool = False,
    ) -> "GeminiLLM":
        """
        Lấy instance singleton của GeminiLLM.

        Tham số:
          - model_name: tên model truyền vào (mặc định gemini-2.0-flash)
          - temperature: tham số nhiệt độ cho LLM
          - force_reload: nếu True sẽ reload lại model ngay cả khi instance đã tồn tại
        """
        if cls._instance is None:
            cls._instance = cls(model_name=model_name, temperature=temperature)
        elif force_reload:
            cls._instance.model_name = model_name
            cls._instance.temperature = temperature
            cls._instance._init_llm()
        return cls._instance

    def invoke(self, messages: List[Dict[str, str]]) -> str:
        if not self.llm:
            raise RuntimeError("LLM chưa được khởi tạo.")
        self.logger.debug(f"Gửi messages: {messages}")
        return self.llm.invoke(messages)

    def reload(self) -> None:
        """Reload lại LLM với cấu hình hiện tại."""
        self._init_llm()
