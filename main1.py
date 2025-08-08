import asyncio
import json
from dataclasses import asdict
from typing import List
from rag.config.chunking_config import ChunkingConfig
from rag.document_parser.pdf_extractor import PdfExtractor
from rag.chunking import Chunking
from core.embedding.embedding_model import EmbeddingModel
from rag.schemas.pdf.text_block import TextBlock
from rag.vector_store import VectorStoreOperations
from rag.embedding import Embedding


collection_name = "689631b0d0ab7efc3956636f"
embedding_dim = 768  # mặc định, sẽ cập nhật khi tạo embedding xong

# Giữ chunks text toàn cục để search
all_chunk_texts: List[str] = []


async def extract_chunk_and_store(pdf_path: str):
    global embedding_dim, all_chunk_texts

    extractor = PdfExtractor(
        enable_text_cleaning=True,
        remove_urls=True,
        remove_page_numbers=True,
        remove_short_lines=True,
        remove_newlines=True,
        remove_references=True,
        remove_stopwords=False,
        min_line_length=3,
        max_block_length=300,
        max_bbox_distance=50.0,
        debug_mode=False,
    )
    embedding_model = await EmbeddingModel.get_instance()
    text_extract = await extractor.extract_all_pages_data(pdf_path)
    blocks: List[TextBlock] = [
        block for page_data in text_extract for block in page_data.blocks
    ]

    chunking_config = ChunkingConfig(
        max_chunk_size=512, min_chunk_size=64, chunk_overlap=200
    )
    chunker = Chunking(config=chunking_config, model_name=embedding_model.model_name)
    chunks = await chunker.chunk_text(blocks)

    # Lưu chunks ra file json
    with open("chunks.json", "w", encoding="utf-8") as f:
        json.dump([asdict(chunk) for chunk in chunks], f, ensure_ascii=False, indent=2)

    # Tạo embedding
    embedding_client = Embedding()
    chunk_texts = [chunk.text for chunk in chunks]
    embeddings = await embedding_client.embed_documents(chunk_texts)
    embedding_dim = len(embeddings[0])
    print(
        f"Tạo được {len(embeddings)} embeddings, mỗi embedding có độ dài {embedding_dim}"
    )

    # Lưu vào vector store
    vector_ops = VectorStoreOperations.get_instance()
    await vector_ops.create_collection(collection_name, vector_size=embedding_dim)
    metadatas = [{"chunk_index": i} for i in range(len(chunk_texts))]
    await vector_ops.store_vectors(
        texts=chunk_texts,
        collection_name=collection_name,
        metadatas=metadatas,
        vector_size=embedding_dim,
    )
    print(f"Đã lưu {len(chunk_texts)} chunks vào collection '{collection_name}'")

    all_chunk_texts = chunk_texts


async def search_interactive():
    vector_ops = VectorStoreOperations.get_instance()

    while True:
        query = input("\nNhập câu hỏi để tìm kiếm (hoặc gõ 'exit' để thoát): ").strip()
        if query.lower() in ("exit", "quit"):
            print("Thoát chức năng tìm kiếm.")
            break
        if not query:
            print("Vui lòng nhập câu hỏi không để trống.")
            continue

        try:
            results = await vector_ops.search(
                query_text=query,
                collection_names=[collection_name],
                top_k=5,
                score_threshold=0.7,
                vector_size=embedding_dim,
            )
            if not results:
                print("Không tìm thấy kết quả nào.")
            else:
                # Gom liền tất cả text, loại bỏ xuống dòng và khoảng trắng thừa trong từng đoạn
                combined_text = "".join(
                    res.text.replace("\n", " ").strip() for res in results
                )

                print(f"Kết quả tìm kiếm cho: '{query}':\n")
                print(combined_text)
        except Exception as e:
            print(f"Lỗi khi tìm kiếm: {e}")


async def main():
    while True:
        print("\n====== MENU ======")
        print("1. Extract + Chunk + Embed + Store")
        print("2. Search interactive")
        print("3. Thoát")

        choice = input("Chọn số (1-3): ").strip()
        if choice == "1":
            pdf_path = input(
                "Nhập đường dẫn file PDF (ví dụ C:/Users/.../text.pdf): "
            ).strip()
            if pdf_path:
                await extract_chunk_and_store(pdf_path)
            else:
                print("Đường dẫn không hợp lệ.")
        elif choice == "2":
            await search_interactive()
        elif choice == "3":
            print("Kết thúc chương trình.")
            break
        else:
            print("Lựa chọn không hợp lệ. Vui lòng chọn lại.")


if __name__ == "__main__":
    asyncio.run(main())
