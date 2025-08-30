import asyncio
import logging
from pathlib import Path

from core.llm import QwenLLM
from rag.parser.pdf_parser import PDFParser
from rag.chunking.chunker import Chunker
from core.embedding import EmbeddingModel
from rag.vector_store.manager import VectorStoreManager
from rag.retrieval.retriever import Retriever
import os
import dotenv

dotenv.load_dotenv()


# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
)
logger = logging.getLogger("test_pdf_pipeline")

# Cấu hình
PDF_PATH = Path("diabetes.pdf")
COLLECTION_NAME = "68b2a701f5cf267abd685844"
DOCUMENT_ID = "diabetes_main_page"
KNOWLEDGE_ID = "diabetes_2025_vn"
EMBEDDING_MODEL = None


# --- HÀM 1: Ingest PDF (Parse → Chunk → Embed → Save) ---
async def ingest_pdf_document():
    """
    Pipeline xử lý PDF: parse → chunk → embed → lưu vào Qdrant
    """
    global EMBEDDING_MODEL

    if not PDF_PATH.exists():
        logger.error(f"❌ File PDF không tồn tại: {PDF_PATH}")
        return

    logger.info(f"📄 Bắt đầu xử lý PDF: {PDF_PATH}")

    # 1. Khởi tạo parser
    parser = PDFParser()

    try:
        # 2. Parse PDF
        logger.info("🔍 Đang parse PDF...")
        parsed = await parser.parse_async(PDF_PATH)
        logger.info(f"✅ Parse thành công. Độ dài nội dung: {len(parsed.content)} ký tự")

        # 3. Chunk
        logger.info("✂️  Đang chunking nội dung...")
        if EMBEDDING_MODEL is None:
            EMBEDDING_MODEL = await EmbeddingModel.get_instance()
        chunker = Chunker(embedding_model=EMBEDDING_MODEL, max_tokens=512, min_tokens=50)
        chunks = await chunker.chunk_async(parsed)
        logger.info(f"✅ Tạo được {len(chunks)} chunks")

        # 4. Tạo embedding
        logger.info("🧠 Đang tạo embedding...")
        texts = [chunk.content for chunk in chunks]
        embeddings = await EMBEDDING_MODEL.embed_batch(texts, max_batch_size=8)
        logger.info(f"✅ Tạo xong {len(embeddings)} embeddings")

        # 5. Tạo payloads
        payloads = []
        for chunk in chunks:
            payloads.append({
                "content": chunk.content,
                "metadata": {
                    "document_id": DOCUMENT_ID,
                    "knowledge_id": KNOWLEDGE_ID,
                    "chunk_index": chunk.metadata.chunk_index,
                    "chunking_strategy": chunk.metadata.chunking_strategy
                },
                "document_is_active": True,
                "metadata": {
                    "is_active": True
                }
            })

        # 6. Lưu vào Qdrant
        vector_store = VectorStoreManager()
        await vector_store.create_collection_async(COLLECTION_NAME, size=768)
        await vector_store.insert_async(
            name=COLLECTION_NAME,
            embeddings=embeddings,
            payloads=payloads
        )

        logger.info(f"✅ Đã lưu {len(chunks)} chunks vào collection '{COLLECTION_NAME}'")

    except Exception as e:
        logger.error(f"❌ Lỗi trong quá trình ingest: {e}", exc_info=True)
        raise


# --- HÀM 2: Search Retrieval ---
async def search_retrieval():
    """
    Test phần retrieval: tạo embedding + tìm kiếm + in kết quả.
    Không phụ thuộc vào parser, chunk, hay ingest.
    """
    global EMBEDDING_MODEL

    try:
        # 1. Khởi tạo embedding model (singleton)
        if EMBEDDING_MODEL is None:
            EMBEDDING_MODEL = await EmbeddingModel.get_instance()
        logger.info("✅ Đã khởi tạo EmbeddingModel")

        # 2. Khởi tạo Retriever
        retriever = Retriever(collections=[COLLECTION_NAME])
        logger.info(f"✅ Đã khởi tạo Retriever cho collection: {COLLECTION_NAME}")

        # 3. Các câu hỏi test
        queries = [
            "Bệnh tiểu đường là gì?",
        ]

        # 4. Tìm kiếm cho từng query
        for query in queries:
            logger.info(f"\n❓ Câu hỏi: {query}")

            # Tạo embedding
            query_vector = await EMBEDDING_MODEL.embed(query)
            logger.debug(f"🧠 Đã tạo embedding (size: {len(query_vector)})")

            # Gọi retrieve
            results = await retriever.retrieve(
                query_vector=query_vector,
                top_k=3,
                score_threshold=0.6,
                document_is_active=True,
                metadata__is_active=True
            )

            # Hiển thị kết quả
            if not results:
                logger.info("  ❌ Không tìm thấy kết quả phù hợp.")
            else:
                for i, r in enumerate(results):
                    content = r["payload"]["content"]
                    score = r["score"]
                    collection = r["collection"]
                    print(f"  [{i+1}] (Score: {score:.3f}) | (Col: {collection})")
                    print(f"      {content[:200]}...")

    except Exception as e:
        logger.error(f"❌ Lỗi trong quá trình search: {e}", exc_info=True)
        raise

async def test_llm():
    llm = QwenLLM(
        model=os.getenv("QWEN_MODEL"),
        base_url=os.getenv("QWEN_URL")
    )
    result = await llm.generate(
        prompt="Tui bị đái tháo đường loại 2 thì nên làm gì",
        temperature=0.7
    )
    print(result)


# --- Hàm main ---
async def main():
    # Bước 1: Ingest
    # await ingest_pdf_document()

    # Bước 2: Search
    await search_retrieval()

    # await test_llm()


if __name__ == "__main__":
    asyncio.run(main())