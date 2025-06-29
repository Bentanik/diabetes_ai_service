"""
Dịch vụ embedding sử dụng HuggingFace models.
Tối ưu hóa cho tiếng Việt và hỗ trợ nhiều models khác nhau.
"""

import os
from typing import List, Optional
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from core.logging_config import get_logger
import torch

logger = get_logger(__name__)


def auto_detect_device() -> str:
    """
    Tự động phát hiện thiết bị tốt nhất có sẵn.

    Returns:
        Device string: "cuda" nếu có GPU, "cpu" nếu không
    """
    if torch.cuda.is_available():
        return "cuda"
    else:
        # Kiểm tra số lõi CPU để optimize
        cpu_count = os.cpu_count() or 4
        logger.info(f"GPU không available, sử dụng CPU với {cpu_count} cores")
        return "cpu"


class Embeddings:
    """
    Dịch vụ embedding HuggingFace được tối ưu cho tiếng Việt.
    Sử dụng sentence transformers với hỗ trợ GPU tự động.
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-base",
        device: str = "auto",
        normalize_embeddings: bool = True,
    ):
        """
        Khởi tạo dịch vụ embedding.

        Args:
            model_name: Tên model HuggingFace
            device: Thiết bị chạy ("auto", "cpu", "cuda")
            normalize_embeddings: Chuẩn hóa vector embeddings
        """
        self.model_name = model_name

        # Auto-detect device nếu được yêu cầu
        if device == "auto":
            device = auto_detect_device()

        self.device = device
        self.normalize_embeddings = normalize_embeddings

        # Cấu hình tối ưu cho CPU nếu cần
        model_kwargs = {"device": device}
        encode_kwargs = {"normalize_embeddings": normalize_embeddings}

        # Tối ưu cho CPU
        if device == "cpu":
            # Tăng số threads cho CPU processing
            torch.set_num_threads(torch.get_num_threads())
            logger.info(f"Tối ưu CPU: sử dụng {torch.get_num_threads()} threads")

            # HuggingFaceEmbeddings không support custom batch_size trong encode_kwargs
            # Chỉ giữ normalize_embeddings

        # Khởi tạo HuggingFace embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

        logger.info(f"Đã khởi tạo Vietnamese embedding service")
        logger.info(f"Model: {model_name}")
        logger.info(f"Device: {device}")
        logger.info(f"Normalize: {normalize_embeddings}")

        if device == "cpu":
            logger.warning("⚠️  ĐANG CHẠY TRÊN CPU - Performance sẽ chậm hơn GPU")
            logger.info("💡 Để tăng tốc: cài PyTorch với CUDA hoặc dùng model nhỏ hơn")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Tạo embeddings cho danh sách văn bản.

        Args:
            texts: Danh sách văn bản cần embedding

        Returns:
            Danh sách vectors embedding
        """
        try:
            embeddings = self.embeddings.embed_documents(texts)
            logger.debug(f"Đã tạo embeddings cho {len(texts)} văn bản")
            return embeddings
        except Exception as e:
            logger.error(f"Lỗi tạo embeddings cho documents: {e}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """
        Tạo embedding cho câu hỏi.

        Args:
            text: Câu hỏi cần embedding

        Returns:
            Vector embedding của câu hỏi
        """
        try:
            embedding = self.embeddings.embed_query(text)
            logger.debug(f"Đã tạo embedding cho query: {text[:50]}...")
            return embedding
        except Exception as e:
            logger.error(f"Lỗi tạo embedding cho query: {e}")
            raise

    def get_info(self) -> dict:
        """Lấy thông tin về service embedding."""
        return {
            "provider": "huggingface",
            "model_name": self.model_name,
            "device": self.device,
            "normalize_embeddings": self.normalize_embeddings,
            "supports_vietnamese": True,
            "model_type": "sentence_transformer",
        }


class MultiEmbeddingService:
    """
    Dịch vụ embedding hỗ trợ nhiều provider.
    Cho phép chuyển đổi giữa HuggingFace và OpenAI.
    """

    def __init__(
        self,
        provider: str = "huggingface",
        model_name: str = "intfloat/multilingual-e5-base",
        api_key: Optional[str] = None,
        device: str = "auto",
    ):
        """
        Khởi tạo multi embedding service.

        Args:
            provider: Loại provider (huggingface hoặc openai)
            model_name: Tên model
            api_key: API key cho OpenAI (nếu cần)
            device: Thiết bị chạy
        """
        self.provider = provider
        self.model_name = model_name
        self.device = device

        if provider == "huggingface":
            self.embeddings = Embeddings(model_name=model_name, device=device)
        elif provider == "openai":
            if not api_key:
                api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key không được cung cấp")

            self.embeddings = OpenAIEmbeddings(
                api_key=api_key, model=model_name or "text-embedding-3-small"
            )
        else:
            raise ValueError(f"Provider không được hỗ trợ: {provider}")

        logger.info(f"Đã khởi tạo multi embedding service với provider: {provider}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Tạo embeddings cho danh sách văn bản."""
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """Tạo embedding cho câu hỏi."""
        return self.embeddings.embed_query(text)

    def get_info(self) -> dict:
        """Lấy thông tin về service."""
        if isinstance(self.embeddings, Embeddings):
            return self.embeddings.get_info()
        else:
            return {
                "provider": self.provider,
                "model_name": self.model_name,
                "device": self.device if self.provider == "huggingface" else "api",
                "supports_vietnamese": self.provider == "huggingface",
            }


# Cấu hình model embedding cố định cho tiếng Việt
MODEL_DEFAULT_EMBEDDING = "intfloat/multilingual-e5-base"

# Không cần configs phức tạp - chỉ dùng E5-base cho mọi trường hợp


def get_embedding_service(
    provider: str = "huggingface",
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    device: str = "auto",
) -> MultiEmbeddingService:
    """
    Factory function để tạo embedding service.

    Args:
        provider: Loại provider (huggingface hoặc openai)
        model_name: Tên model (mặc định: intfloat/multilingual-e5-base)
        api_key: API key cho OpenAI
        device: Thiết bị chạy ("auto", "cpu", "cuda")

    Returns:
        Instance của MultiEmbeddingService
    """
    # Auto-detect device
    if device == "auto":
        device = auto_detect_device()

    # Sử dụng model cố định E5-base cho tất cả trường hợp
    if not model_name:
        if provider == "huggingface":
            model_name = MODEL_DEFAULT_EMBEDDING
        elif provider == "openai":
            model_name = "text-embedding-3-small"
        else:
            model_name = MODEL_DEFAULT_EMBEDDING

    # Ensure model_name is not None
    final_model_name = model_name or "intfloat/multilingual-e5-base"

    return MultiEmbeddingService(
        provider=provider, model_name=final_model_name, api_key=api_key, device=device
    )


def get_vietnamese_embedding_service(
    model_name: str = "intfloat/multilingual-e5-base", device: str = "cpu"
) -> Embeddings:
    """
    Tạo embedding service tối ưu cho tiếng Việt.

    Args:
        model_name: Tên model HuggingFace
        device: Thiết bị chạy

    Returns:
        Embeddings instance
    """
    return Embeddings(model_name=model_name, device=device)
