# main.py
import asyncio
import json
import logging
from pathlib import Path

# Giả sử các module ở đúng vị trí
from rag.parser.pdf_parser import PDFParser
from rag.chunking.chunker import Chunker
from core.embedding import EmbeddingModel
from rag.dataclasses import ParsedContent


# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    # === 1. Đường dẫn file ===
    pdf_path = "diabetes.pdf"
    output_json = "output_chunks.json"

    if not Path(pdf_path).exists():
        logger.error(f"File không tồn tại: {pdf_path}")
        return

    logger.info(f"Bắt đầu xử lý file: {pdf_path}")

    try:
        # === 2. Khởi tạo các thành phần ===
        parser = PDFParser()
        embedding_model = await EmbeddingModel.get_instance()

        chunker = Chunker(
            embedding_model=embedding_model,
            max_tokens=512,
            min_tokens=100,
            overlap_tokens=64,
            similarity_threshold=0.6
        )

        logger.info("🔄 Đang parse PDF...")
        parsed_content = await parser.parse_async(pdf_path)

        logger.info(f"✅ Parse thành công: {parsed_content.metadata['num_pages']} trang, {len(parsed_content.tables)} bảng")

        logger.info("✂️  Đang chia nhỏ văn bản theo ngữ nghĩa...")
        chunks = await chunker.chunk_async(parsed_content)

        logger.info(f"✅ Tạo được {len(chunks)} chunk")

        # === 5. (Tùy chọn) Tạo embedding cho từng chunk ===
        # Nếu bạn muốn lưu luôn embedding
        generate_embeddings = False  # Đặt True nếu muốn
        chunk_dicts = []

        for i, chunk in enumerate(chunks):
            chunk_data = {
                "chunk_index": chunk.metadata.chunk_index,
                "content": chunk.content,
                "word_count": chunk.metadata.word_count,
                "chunking_strategy": chunk.metadata.chunking_strategy
            }

            if generate_embeddings:
                try:
                    emb = await embedding_model.embed(chunk.content)
                    chunk_data["embedding"] = emb
                except Exception as e:
                    logger.warning(f"Embedding failed for chunk {i}: {e}")
                    chunk_data["embedding"] = None

            chunk_dicts.append(chunk_data)

        # === 6. Metadata tổng hợp ===
        result = {
            "source_file": parsed_content.file_path,
            "file_type": parsed_content.file_type,
            "metadata": parsed_content.metadata,
            "tables_extracted": len(parsed_content.tables),
            "total_chunks": len(chunk_dicts),
            "chunks": chunk_dicts
        }

        # === 7. Lưu ra file JSON ===
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ Đã lưu kết quả vào: {output_json}")

        # === 8. In thử 2 chunk đầu ===
        print("\n--- Ví dụ chunk đầu tiên ---")
        if chunks:
            print(chunks[0].content[:500] + ("..." if len(chunks[0].content) > 500 else ""))
        else:
            print("(Không có chunk nào)")

    except Exception as e:
        logger.error(f"❌ Lỗi xử lý: {str(e)}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())