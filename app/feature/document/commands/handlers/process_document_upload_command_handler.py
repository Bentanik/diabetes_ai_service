"""
Process Document Upload Command Handler - Xử lý command upload tài liệu

File này định nghĩa handler để xử lý ProcessDocumentUploadCommand, thực hiện việc upload tài liệu vào database và storage.
"""

import os
import tempfile
from bson import ObjectId
from app.database import get_collections
from app.database.enums import DocumentJobStatus, DocumentType
from app.database.models import (
    DocumentJobModel,
    DocumentModel,
    DocumentParserModel,
)
from app.database.value_objects import DocumentFile, PageLocation, BoundingBox
from app.storage import MinioManager
from core.cqrs import CommandHandler, CommandRegistry
from shared.messages import DocumentResult
from rag.document_parser import PdfExtractor

from ..process_document_upload_command import ProcessDocumentUploadCommand

from core.result import Result
from utils import get_logger, async_analyze_diabetes_content, FileHashUtils


@CommandRegistry.register_handler(ProcessDocumentUploadCommand)
class ProcessDocumentUploadCommandHandler(CommandHandler):
    def __init__(self):
        super().__init__()
        self.logger = get_logger(__name__)
        self.collections = get_collections()
        self.minio_manager = MinioManager.get_instance()
        self.pdf_extractor = PdfExtractor(
            enable_text_cleaning=True,
            remove_urls=True,
            remove_page_numbers=True,
            remove_short_lines=True,
            min_line_length=3,
        )

    async def execute(self, command: ProcessDocumentUploadCommand) -> Result[None]:
        self.logger.info(f"Xử lý tài liệu upload: {command.document_job_id}")

        temp_path = None
        temp_dir = None

        try:
            # Tìm document job trước
            document_job_data = await self.collections.document_jobs.find_one(
                {
                    "_id": ObjectId(command.document_job_id),
                }
            )
            if document_job_data is None:
                self.logger.info(
                    f"Không tìm thấy document job với ID: {command.document_job_id}"
                )
                return Result.failure(
                    message=DocumentResult.NOT_FOUND.message,
                    code=DocumentResult.NOT_FOUND.code,
                )

            document_job = DocumentJobModel.from_dict(document_job_data)
            if document_job is None:
                return Result.failure(
                    message=DocumentResult.NOT_FOUND.message,
                    code=DocumentResult.NOT_FOUND.code,
                )

            # Cập nhật trạng thái processing
            await self._update_document_job(
                command.document_job_id,
                status=DocumentJobStatus.PROCESSING,
                progress=15,
                progress_message="Đang xử lý tài liệu",
            )

            # Tải tài liệu về
            await self._update_document_job(
                command.document_job_id,
                status=DocumentJobStatus.PROCESSING,
                progress=30,
                progress_message="Đang tải tài liệu về",
            )
            temp_dir, temp_path = await self._download_file_to_temp(
                document_job.file_path
            )

            # Tính hash và kiểm tra trùng lặp
            await self._update_document_job(
                command.document_job_id,
                status=DocumentJobStatus.PROCESSING,
                progress=35,
                progress_message="Đang kiểm tra trùng lặp tài liệu",
            )

            file_hash = FileHashUtils.calculate_file_hash(temp_path)
            self.logger.info(f"File hash: {file_hash}")

            # Kiểm tra trùng lặp trong knowledge base
            duplicate_check = await self._check_duplicate_document(
                knowledge_id=document_job.knowledge_id, file_hash=file_hash
            )

            if duplicate_check:
                self.logger.warning(
                    f"Tài liệu trùng lặp được phát hiện: {duplicate_check['title']}"
                )
                await self._update_document_job(
                    command.document_job_id,
                    status=DocumentJobStatus.FAILED,
                    progress_message=f"Tài liệu trùng lặp với: {duplicate_check['title']}",
                )
                return Result.failure(
                    message=DocumentResult.DUPLICATE.message,
                    code=DocumentResult.DUPLICATE.code,
                )

            # Làm sạch dữ liệu
            await self._update_document_job(
                command.document_job_id,
                status=DocumentJobStatus.PROCESSING,
                progress=50,
                progress_message="Đang trích xuất và làm sạch nội dung",
            )

            cleaned_text, pages_data = await self._extract_and_clean_text(temp_path)

            if not cleaned_text:
                self.logger.warning(f"Không thể extract text từ file: {temp_path}")
                cleaned_text = ""

            # Tính điểm diabetes
            await self._update_document_job(
                command.document_job_id,
                status=DocumentJobStatus.PROCESSING,
                progress=65,
                progress_message="Đang phân tích nội dung diabetes",
            )

            diabetes_score = await self._calculate_diabetes_score(
                cleaned_text
            )

            self.logger.info(
                f"Điểm diabetes cho document {command.document_job_id}: {diabetes_score})"
            )

            # Lưu tài liệu vào database
            await self._update_document_job(
                command.document_job_id,
                status=DocumentJobStatus.PROCESSING,
                progress=80,
                progress_message="Đang lưu tài liệu vào database",
            )

            file_size = os.path.getsize(temp_path)
            document_model = DocumentModel(
                knowledge_id=document_job.knowledge_id,
                title=document_job.title,
                description=document_job.description,
                type=DocumentType.UPLOAD,
                priority_diabetes=diabetes_score,
                file=DocumentFile(
                    path=document_job.file_path,
                    size_bytes=file_size,
                    hash=file_hash,
                ),
            )

            document_model.id = document_job.document_id

            await self.collections.documents.insert_one(document_model.to_dict())

            # Lưu document parser results
            await self._update_document_job(
                command.document_job_id,
                status=DocumentJobStatus.PROCESSING,
                progress=90,
                progress_message="Đang lưu kết quả phân tích nội dung",
            )

            await self._save_document_parser_results(
                document_id=str(document_model.id),
                knowledge_id=document_job.knowledge_id,
                pages_data=pages_data,
                cleaned_text=cleaned_text,
            )

            # Cập nhật knowledge stats
            await self._update_document_job(
                command.document_job_id,
                status=DocumentJobStatus.PROCESSING,
                progress=95,
                progress_message="Đang cập nhật thống kê cơ sở tri thức",
            )

            await self._update_knowledge_stats(
                knowledge_id=document_job.knowledge_id, file_size=file_size
            )

            # Cập nhật document job với kết quả diabetes
            await self._update_document_job(
                command.document_job_id,
                status=DocumentJobStatus.COMPLETED,
                progress=100,
                progress_message="Đã hoàn thành phân tích và lưu tài liệu",
                priority_diabetes=diabetes_score,
            )

            return Result.success(
                message=DocumentResult.CREATED.message,
                code=DocumentResult.CREATED.code,
            )

        except Exception as e:
            self.logger.error(f"Lỗi xử lý tài liệu upload: {e}")
            await self._update_document_job(
                command.document_job_id,
                status=DocumentJobStatus.FAILED,
                progress_message=f"Lỗi: {str(e)}",
            )
            return Result.failure(
                message=DocumentResult.UPLOAD_FAILED.message,
                code=DocumentResult.UPLOAD_FAILED.code,
            )

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            if temp_dir and os.path.exists(temp_dir):
                os.rmdir(temp_dir)

    async def _check_duplicate_document(
        self, knowledge_id: str, file_hash: str
    ) -> dict:
        """
        Kiểm tra tài liệu trùng lặp trong knowledge base

        Args:
            knowledge_id: ID của knowledge base
            file_hash: Hash của file cần kiểm tra

        Returns:
            dict: Thông tin document trùng lặp hoặc None
        """
        try:
            duplicate = await self.collections.documents.find_one(
                {"knowledge_id": knowledge_id, "file_hash": file_hash}
            )

            if duplicate:
                return {
                    "id": str(duplicate["_id"]),
                    "title": duplicate.get("title", "Unknown"),
                    "created_at": duplicate.get("created_at"),
                }

            return None

        except Exception as e:
            self.logger.error(f"Lỗi kiểm tra duplicate document: {e}")
            return None

    async def _update_knowledge_stats(self, knowledge_id: str, file_size: int) -> None:
        """
        Cập nhật thống kê cho knowledge base

        Args:
            knowledge_id: ID của knowledge base
            file_size: Kích thước file được thêm
        """
        try:
            # Tăng document count và total size
            await self.collections.knowledges.update_one(
                {"_id": ObjectId(knowledge_id)},
                {"$inc": {"document_count": 1, "total_size_bytes": file_size}},
            )

            self.logger.info(
                f"Đã cập nhật stats cho knowledge {knowledge_id}: +1 document, +{file_size} bytes"
            )

        except Exception as e:
            self.logger.error(f"Lỗi cập nhật knowledge stats: {e}")
            # Không raise exception để không làm fail toàn bộ process

    async def _extract_and_clean_text(self, file_path: str) -> tuple[str, list]:
        """
        Trích xuất và làm sạch text từ file

        Args:
            file_path: Đường dẫn file cần xử lý

        Returns:
            tuple[str, list]: (cleaned_text, pages_data)
        """
        try:
            if file_path.lower().endswith(".pdf"):
                # Extract text blocks với cleaning enabled
                pages_data = self.pdf_extractor.extract_all_pages_data(file_path)

                if not pages_data:
                    self.logger.warning(
                        f"Không extract được pages từ file: {file_path}"
                    )
                    return "", []

                # Kết hợp tất cả text blocks từ tất cả pages
                all_text_blocks = []
                for page_data in pages_data:
                    for block in page_data.blocks:
                        if block.context.strip():  # block.context là text đã cleaned
                            all_text_blocks.append(block.context)

                if not all_text_blocks:
                    self.logger.warning(
                        f"Không có text blocks sau khi clean: {file_path}"
                    )
                    return "", pages_data

                # Kết hợp tất cả text
                full_text = " ".join(all_text_blocks)

                self.logger.info(
                    f"Extracted và cleaned {len(all_text_blocks)} text blocks, total length: {len(full_text)} chars"
                )

                return full_text, pages_data

            else:
                self.logger.warning(f"File type không được hỗ trợ: {file_path}")
                return "", []

        except Exception as e:
            self.logger.error(f"Lỗi extract và clean text: {e}")
            return "", []

    async def _save_document_parser_results(
        self, document_id: str, knowledge_id: str, pages_data: list, cleaned_text: str
    ) -> None:
        """
        Lưu kết quả phân tích document vào DocumentParserModel

        Args:
            document_id: ID của document
            knowledge_id: ID của knowledge
            pages_data: Dữ liệu pages từ PDF extractor
            cleaned_text: Text đã được làm sạch
        """
        try:
            parser_records = []

            if pages_data:
                # Tạo DocumentParserModel cho từng text block
                for page_idx, page_data in enumerate(pages_data):
                    for block_idx, block in enumerate(page_data.blocks):
                        if block.context.strip():
                            # Tạo BoundingBox từ block metadata
                            bbox_data = (
                                block.metadata.bbox
                                if hasattr(block.metadata, "bbox")
                                else {}
                            )
                            bbox = BoundingBox(
                                x0=bbox_data.get("left", 0.0),
                                y0=bbox_data.get("top", 0.0),
                                x1=bbox_data.get("right", 0.0),
                                y1=bbox_data.get("bottom", 0.0),
                            )

                            # Tạo PageLocation
                            location = PageLocation(
                                page=page_idx,
                                bbox=bbox,
                                block_index=block_idx,
                                doc_type=DocumentType.UPLOAD,
                            )

                            # Tạo DocumentParserModel
                            parser_model = DocumentParserModel(
                                document_id=document_id,
                                knowledge_id=knowledge_id,
                                content=block.context,
                                location=location,
                                is_active=True,
                            )

                            parser_records.append(parser_model.to_dict())

                # Batch insert tất cả records
                if parser_records:
                    await self.collections.document_parsers.insert_many(parser_records)
                    self.logger.info(
                        f"Đã lưu {len(parser_records)} document parser records cho document {document_id}"
                    )
                else:
                    self.logger.warning(
                        f"Không có parser records để lưu cho document {document_id}"
                    )

            else:
                # Fallback: tạo một record duy nhất với toàn bộ cleaned text
                if cleaned_text.strip():
                    bbox = BoundingBox(x0=0.0, y0=0.0, x1=0.0, y1=0.0)
                    location = PageLocation(
                        source="full_document",
                        page=0,
                        bbox=bbox,
                        block_index=0,
                        doc_type=DocumentType.UPLOAD,
                    )

                    parser_model = DocumentParserModel(
                        document_id=document_id,
                        content=cleaned_text,
                        location=location,
                        is_active=True,
                    )

                    await self.collections.document_parsers.insert_one(
                        parser_model.to_dict()
                    )
                    self.logger.info(
                        f"Đã lưu fallback parser record cho document {document_id}"
                    )

        except Exception as e:
            self.logger.error(f"Lỗi lưu document parser results: {e}")

    async def _calculate_diabetes_score(self, cleaned_text: str) -> float:
        """
        Tính điểm diabetes cho text đã được làm sạch

        Args:
            cleaned_text: Text đã được extract và làm sạch

        Returns:
            tuple[float, bool]: (diabetes_score)
        """
        try:
            if not cleaned_text or not cleaned_text.strip():
                self.logger.warning("Text rỗng, không thể tính điểm diabetes")
                return 0.0, False

            # Nếu text quá dài, lấy mẫu để phân tích
            if len(cleaned_text) > 10000:
                # Lấy phần đầu, giữa và cuối
                start_text = cleaned_text[:3000]
                mid_point = len(cleaned_text) // 2
                mid_text = cleaned_text[mid_point - 1500 : mid_point + 1500]
                end_text = cleaned_text[-3000:]

                sample_text = f"{start_text} {mid_text} {end_text}"
                self.logger.info(
                    f"Text quá dài ({len(cleaned_text)} chars), sử dụng sample {len(sample_text)} chars"
                )
            else:
                sample_text = cleaned_text

            analysis = await async_analyze_diabetes_content(sample_text)

            self.logger.info(
                f"Diabetes analysis - Score: {analysis.final_score}, "
                f"Level: {analysis.relevance_level}, "
                f"Semantic: {analysis.semantic_score}, "
                f"Keyword: {analysis.keyword_score}, "
                f"Words: {analysis.word_count}"
            )

            return analysis.final_score

        except Exception as e:
            self.logger.error(f"Lỗi tính điểm diabetes: {e}")
            return 0.0, False

    async def _update_document_job(
        self,
        document_job_id: str,
        status: DocumentJobStatus = None,
        progress: float = None,
        progress_message: str = None,
        priority_diabetes: float = None,
    ) -> None:

        update_fields = {
            k: v
            for k, v in {
                "status": status,
                "progress": progress,
                "progress_message": progress_message,
                "priority_diabetes": priority_diabetes,
            }.items()
            if v is not None
        }

        if update_fields:
            await self.collections.document_jobs.update_one(
                {"_id": ObjectId(document_job_id)},
                {"$set": update_fields},
            )

    async def _download_file_to_temp(self, file_path: str) -> tuple[str, str]:
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, os.path.basename(file_path))

        bucket, object_path = file_path.split("/", 1)
        response = self.minio_manager.get_file(
            bucket_name=bucket, object_name=object_path
        )

        with open(temp_path, "wb") as f:
            for chunk in response.stream(32 * 1024):
                f.write(chunk)

        return temp_dir, temp_path
