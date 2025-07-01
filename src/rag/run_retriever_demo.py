"""
Demo script để chạy thử Vietnamese Retriever
"""

import asyncio
import logging
from typing import List
from langchain.schema import Document

from aiservice.src.rag.retriever import Retriever
from aiservice.src.rag.embedding import create_vietnamese_optimized_embeddings
from aiservice.src.rag.vector_store import create_vietnamese_vector_store

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_documents() -> List[Document]:
    """Tạo một số documents mẫu để test."""
    return [
        Document(
            page_content="Hà Nội là thủ đô của Việt Nam, một thành phố với hơn 1000 năm lịch sử.",
            metadata={"source": "wiki", "topic": "địa lý", "id": "doc1"},
        ),
        Document(
            page_content="Phở là một món ăn truyền thống của Việt Nam, được làm từ bánh phở, thịt bò hoặc gà.",
            metadata={"source": "food_guide", "topic": "ẩm thực", "id": "doc2"},
        ),
        Document(
            page_content="Việt Nam có 54 dân tộc anh em, trong đó dân tộc Kinh chiếm đa số.",
            metadata={"source": "wiki", "topic": "văn hóa", "id": "doc3"},
        ),
        Document(
            page_content="Áo dài là trang phục truyền thống của người Việt Nam, đặc biệt phổ biến trong các dịp lễ tết.",
            metadata={"source": "culture_guide", "topic": "văn hóa", "id": "doc4"},
        ),
        Document(
            page_content="Hạ Long là vịnh biển nổi tiếng của Việt Nam, được UNESCO công nhận là di sản thiên nhiên thế giới.",
            metadata={"source": "travel_guide", "topic": "du lịch", "id": "doc5"},
        ),
    ]


async def run_retriever_demo():
    """Chạy demo retriever với một số câu query mẫu."""
    try:
        # Khởi tạo embedding service
        embedding_service = create_vietnamese_optimized_embeddings()

        # Khởi tạo vector store với documents mẫu
        vector_store = create_vietnamese_vector_store(
            embeddings=embedding_service.get_langchain_embeddings()
        )

        # Thêm documents mẫu vào vector store
        sample_docs = create_sample_documents()
        for doc in sample_docs:
            vector_store.add_documents([doc])

        # Khởi tạo retriever
        retriever = Retriever(
            embedding_service=embedding_service,
            vector_store=vector_store,
            top_k=3,
            score_threshold=0.5,
        )

        # Danh sách câu query để test
        test_queries = [
            "Thủ đô Việt Nam có gì đặc biệt?",
            "Món ăn truyền thống Việt Nam?",
            "Di sản thiên nhiên nổi tiếng ở Việt Nam?",
            "Trang phục truyền thống Việt Nam?",
            "Các dân tộc ở Việt Nam?",
        ]

        # Chạy test với từng câu query
        for query in test_queries:
            logger.info(f"\nQuery: {query}")

            # Lấy documents liên quan
            docs = await retriever.aget_relevant_documents(query)

            # In kết quả
            logger.info(f"Tìm thấy {len(docs)} documents liên quan:")
            for i, doc in enumerate(docs, 1):
                logger.info(f"\n{i}. Content: {doc.page_content}")
                logger.info(f"   Score: {doc.metadata.get('score', 'N/A'):.3f}")
                logger.info(f"   Source: {doc.metadata.get('source', 'N/A')}")
                logger.info(f"   Topic: {doc.metadata.get('topic', 'N/A')}")

    except Exception as e:
        logger.error(f"Error running demo: {e}", exc_info=True)


if __name__ == "__main__":
    # Chạy demo
    asyncio.run(run_retriever_demo())
