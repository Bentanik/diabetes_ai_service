"""Client LLM cho OpenRouter."""

from langchain_community.chat_models import ChatOpenAI
from config.settings import get_api_key, get_config
from core.exceptions import ServiceError
from core.logging_config import get_logger

logger = get_logger(__name__)


# Client LLM
class LLMClient:
    """Client LLM sử dụng OpenRouter."""

    def __init__(self):
        try:
            api_key = get_api_key()
            self.client = ChatOpenAI(
                model=get_config("default_model")
                or "meta-llama/llama-3.3-8b-instruct:free",
                temperature=get_config("default_temperature") or 0.3,
                api_key=api_key,
                base_url=get_config("openrouter_base_url")
                or "https://openrouter.ai/api/v1",
            )
            logger.info("Khởi tạo LLM client thành công")
        except Exception as e:
            logger.error(f"Khởi tạo LLM client thất bại: {e}")
            raise ServiceError(f"Không thể khởi tạo LLM client: {e}")

    async def generate(self, prompt: str) -> str:
        """Tạo phản hồi từ LLM."""
        try:
            logger.debug(f"Đang tạo phản hồi cho prompt có độ dài: {len(prompt)}")
            response = await self.client.ainvoke(prompt)

            # Trích xuất text từ response
            if hasattr(response, "content"):
                result = str(response.content)
            else:
                result = str(response)

            logger.debug(f"Độ dài phản hồi được tạo: {len(result)}")
            return result

        except Exception as e:
            logger.error(f"Tạo phản hồi LLM thất bại: {e}")
            raise ServiceError(f"Tạo phản hồi LLM thất bại: {e}")


# Instance toàn cục
_llm_client = None


def get_llm():
    """Lấy instance LLM client toàn cục."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


# Các hàm để tương thích với code cũ
async def query_care_plan_llm(prompt: str) -> str:
    """Tạo phản hồi kế hoạch chăm sóc."""
    return await get_llm().generate(prompt)


async def query_note_record_llm(prompt: str) -> str:
    """Tạo phản hồi phân tích kết quả đo."""
    return await get_llm().generate(prompt)
