import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from utils import get_logger
from typing import Optional

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise EnvironmentError(
        "GOOGLE_API_KEY chưa được thiết lập! Hãy thêm vào file .env hoặc biến môi trường hệ thống."
    )


logger = get_logger(__name__)

_gemini_instance: Optional[ChatGoogleGenerativeAI] = None


def get_gemini_llm() -> ChatGoogleGenerativeAI:
    global _gemini_instance
    if _gemini_instance is None:
        try:
            logger.info("Đang khởi tạo Gemini LLM...")
            _gemini_instance = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0.2,
                convert_system_message_to_human=True,
            )
            logger.info("Khởi tạo Gemini LLM thành công.")
        except Exception as e:
            logger.error(f"Lỗi khởi tạo Gemini LLM: {str(e)}", exc_info=True)
            raise
    return _gemini_instance
