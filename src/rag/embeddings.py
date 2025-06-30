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


def get_device() -> str:
    """
    Chọn thiết bị tính toán: 'cuda' nếu GPU khả dụng, 'cpu' nếu không.

    Returns:
        str: Tên thiết bị.
    """
    if torch.cuda.is_available():
        logger.info("GPU khả dụng, sử dụng 'cuda'.")
        return "cuda"
    cpu_count = os.cpu_count() or 1
    logger.info(f"GPU không khả dụng, sử dụng 'cpu' ({cpu_count} cores).")
    return "cpu"


class Embeddings:
    """
    Service embedding HuggingFace.
    Sử dụng sentence transformers, hỗ trợ auto device.
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-base",
        device: str = "auto",
        normalize_embeddings: bool = True,
    ):
        """
        Khởi tạo Embedding service.

        Args:
            model_name: Tên model.
            device: 'auto', 'cpu', hoặc 'cuda'.
            normalize_embeddings: Chuẩn hóa vector.
        """
        self.model_name = model_name

        if device == "auto":
            device = get_device()

        self.device = device
        self.normalize_embeddings = normalize_embeddings

        # Cấu hình tối ưu cho CPU nếu cần
        model_kwargs = {"device": device}
        encode_kwargs = {"normalize_embeddings": normalize_embeddings}

        if device == "cpu":
            threads = torch.get_num_threads()
            logger.info(f"Đang chạy trên CPU với {threads} threads.")
        else:
            logger.info(f"Đang chạy trên GPU.")

        # Khởi tạo Embedding
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

        logger.info(f"Đã khởi tạo Vietnamese embedding service")
        logger.info(f"Model: {model_name}")
        logger.info(f"Device: {device}")
        logger.info(f"Normalize: {normalize_embeddings}")

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
        Tạo embedding cho query.

        Args:
            query: Câu truy vấn cần embedding.

        Returns:
            Vector embedding của query.
        """
        try:
            embedding = self.embeddings.embed_query(text)
            logger.debug(f"Đã tạo embedding cho query: {text[:50]}...")
            return embedding
        except Exception as e:
            logger.error(f"Lỗi tạo embedding cho query: {e}")
            raise

    def get_info(self) -> dict:
        """
        Lấy thông tin service embedding.

        Returns:
            dict: Thông tin chi tiết service.
        """
        return {
            "provider": "huggingface",
            "model_name": self.model_name,
            "device": self.device,
            "normalize_embeddings": self.normalize_embeddings,
            "supports_vietnamese": True,
            "model_type": "sentence_transformer",
        }
