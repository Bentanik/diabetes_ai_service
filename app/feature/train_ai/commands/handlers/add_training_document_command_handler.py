from rag.chunking import Chunk, get_chunking_instance
import json
from app.database import get_collections
from app.database.models import DocumentParserModel
from core.cqrs import CommandHandler, CommandRegistry
from core.result import Result
from ..add_training_document_command import AddTrainingDocumentCommand
from shared.messages import DocumentResult
from utils import get_logger


@CommandRegistry.register_handler(AddTrainingDocumentCommand)
class AddTrainingDocumentCommandHandler(CommandHandler):
    def __init__(self):
        super().__init__()
        self.logger = get_logger(__name__)
        self.collections = get_collections()

    async def execute(self, command: AddTrainingDocumentCommand) -> Result[None]:
        self.logger.info(f"Thêm tài liệu vào vector database: {command.document_id}")

        try:
            # Khởi tạo chunker tại đây vì cần await
            chunking = await get_chunking_instance()

            # Tìm các document parser liên quan
            document_parsers = await self.collections.document_parsers.find(
                {"document_id": command.document_id}
            ).to_list(length=None)

            if not document_parsers:
                self.logger.info(
                    f"Không tìm thấy document với ID: {command.document_id}"
                )
                return Result.failure(
                    message=DocumentResult.NOT_FOUND.message,
                    code=DocumentResult.NOT_FOUND.code,
                )

            document_parsers = [
                DocumentParserModel.from_dict(doc) for doc in document_parsers
            ]

            # Chunk từng document
            all_chunks: list[Chunk] = []
            for parser in document_parsers:
                chunks = await chunking.chunk_text(
                    text=parser.content,
                    metadata={
                        "document_parser_id": parser.id,
                        "is_active": parser.is_active,
                    },
                )
                all_chunks.extend(chunks)

            # Lưu all_chunks vào file JSON
            try:
                json_output = {
                    "document_id": command.document_id,
                    "chunk_count": len(all_chunks),
                    "chunks": all_chunks,
                }
                output_file = f"chunks_{command.document_id}.json"
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(json_output, f, ensure_ascii=False, indent=2)
                self.logger.info(
                    f"Đã chunk và lưu {len(all_chunks)} đoạn văn bản vào {output_file}"
                )
            except Exception as e:
                self.logger.error(f"Lỗi khi lưu chunks vào JSON: {e}")
                # Không làm thất bại command chính, chỉ log lỗi

            return Result.success(
                message=DocumentResult.CREATED.message,
                code=DocumentResult.CREATED.code,
            )

        except Exception as e:
            self.logger.error(f"Lỗi xử lý tài liệu huấn luyện: {e}")
            return Result.failure(
                message=DocumentResult.TRAINING_FAILED.message,
                code=DocumentResult.TRAINING_FAILED.code,
            )
