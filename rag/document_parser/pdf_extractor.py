import json
import logging
import re
from typing import Dict, List, Tuple
import fitz
from ..common import LanguageDetector, MultilingualTokenizer
from .text_filter import TextFilter
from ..schemas.pdf import (
    TextBlock,
    BlockMetadata,
    PageData,
    PageSize,
    BBox
)

logger = logging.getLogger(__name__)

class PdfExtractor:
    """
    PDF Text Extractor sử dụng PyMuPDF với chiến lược phân tích cấu trúc
    """

    HEADING_PATTERNS = [
        r"^(?:CHƯƠNG|PHẦN|MỤC|BÀI|TIẾT|PHỤLỤC|PHỤ LỤC|CHAPTER|SECTION|PART|ARTICLE|APPENDIX)\s+[IVXLCDM\d]+(?:\s|$)",
        r"^(?:\d+|[IVXLCDM]+)\.\s+[A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐĐ]",
        r"^#{1,6}\s+",
    ]

    LIST_PATTERNS = [
        r"^(?:[-•*]|\d+[\.\)]|[a-zA-Z][\.\)]|[IVXLCDM]+[\.\)])\s+",
    ]

    CODE_PATTERNS = [
        r"^```",
        r"^(def |class |import |function |var |const | let )",
    ]

    PROTECTED_NUMBER_PATTERNS = [
        r"\b(?:loại|type|phiên bản|version|số|number|mã|code|id|ID)\s+\d+\b",
        r"\b\d+\s*(?:mg|kg|ml|l|g|%|độ|°C|°F|mm|cm|m|km|Hz|kHz|MHz|GHz)\b",
        r"\b(?:bước|step|giai đoạn|phase|cấp|level|tầng|floor)\s+\d+\b",
        r"\b(?:trang|page|chương|chapter|phần|part|mục|section)\s+\d+\b",
        r"\b(?:năm|year|tháng|month|ngày|day|tuần|week)\s+\d+\b",
        r"\b\d+\s*(?:lần|times|lượt|turn|đợt|batch)\b",
        r"\b(?:model|mô hình|phương pháp|method)\s+\d+\b",
        r"\b\d+(?:\.\d+)?\s*(?:triệu|million|tỷ|billion|nghìn|thousand)\b",
        r"\b(?:covid|COVID)[-\s]*\d+\b",
        r"\b(?:vitamin|Vitamin)\s*[A-Z]\d*\b",
        r"\b\d+\s*(?:chiều|dimension|D)\b",
    ]

    REFERENCE_PATTERNS = [
        r"\[\d+\]",  # [1], [2], [123]
        r"\[\d+[-,\s]*\d*\]",  # [1-3], [1, 2], [1 2]
        r"\(\d+\)",  # (1), (2)
        r"\(\d+[-,\s]*\d*\)",  # (1-3), (1, 2)
        r"\b(?:ref|Ref|REF)\.?\s*\[\d+\]",  # ref[1], Ref.[2]
        r"\b(?:see|See|xem|Xem)\s*\[\d+\]",  # see[1], xem[2]
        r"^\[\d+\]\s*",  # [1] ở đầu dòng
    ]

    def __init__(
        self,
        enable_text_cleaning: bool = True,
        remove_urls: bool = True,
        remove_page_numbers: bool = True,
        remove_short_lines: bool = True,
        remove_newlines: bool = True,
        remove_references: bool = True,
        remove_stopwords: bool = True,
        min_line_length: int = 3,
        max_block_length: int = 300,
        max_bbox_distance: float = 50.0,
        debug_mode: bool = False,
    ):
        self.enable_text_cleaning = enable_text_cleaning
        self.remove_urls = remove_urls
        self.remove_page_numbers = remove_page_numbers
        self.remove_short_lines = remove_short_lines
        self.remove_newlines = remove_newlines
        self.remove_references = remove_references
        self.remove_stopwords = remove_stopwords
        self.min_line_length = min_line_length
        self.max_block_length = max_block_length
        self.max_bbox_distance = max_bbox_distance
        self.debug_mode = debug_mode

        try:
            self.text_detector = LanguageDetector()
        except Exception as e:
            logger.error("Không thể khởi tạo LanguageDetector: %s", e)
            self.text_detector = None

        try:
            self.tokenizer = MultilingualTokenizer()
        except Exception as e:
            logger.error("Không thể khởi tạo MultilingualTokenizer: %s", e)
            self.tokenizer = None

    def _protect_important_numbers(self, text: str) -> Tuple[str, Dict[str, str]]:
        """Bảo vệ các số quan trọng trong text"""
        protected_text = text
        placeholders = {}
        placeholder_counter = 0

        for pattern in self.PROTECTED_NUMBER_PATTERNS:
            matches = list(re.finditer(pattern, protected_text, re.IGNORECASE))
            for match in reversed(matches):
                placeholder = f"__PROTECTED_NUM_{placeholder_counter}__"
                placeholders[placeholder] = match.group()
                protected_text = (
                    protected_text[: match.start()]
                    + placeholder
                    + protected_text[match.end() :]
                )
                placeholder_counter += 1

        if self.debug_mode and placeholders:
            logger.debug("Protected numbers: %s", placeholders)

        return protected_text, placeholders

    def _restore_protected_numbers(
        self, text: str, placeholders: Dict[str, str]
    ) -> str:
        """Khôi phục các số đã được bảo vệ"""
        restored_text = text
        for placeholder, original in placeholders.items():
            restored_text = restored_text.replace(placeholder, original)
        return restored_text

    def _remove_reference_numbers(self, text: str) -> str:
        """Xóa các reference numbers và citations"""
        if not self.remove_references:
            return text

        cleaned_text = text
        for pattern in self.REFERENCE_PATTERNS:
            cleaned_text = re.sub(pattern, "", cleaned_text, flags=re.IGNORECASE)

        return cleaned_text

    def _remove_invisible_unicode(self, text: str) -> str:
        # Xóa các ký tự Unicode ẩn thường gặp khi extract PDF
        return re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)

    def _normalize_whitespace_and_newlines(self, text: str) -> str:
        if not text:
            return text

        # Xóa ký tự Unicode ẩn trước
        text = self._remove_invisible_unicode(text)
        text = re.sub(r"\n{2,}", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = text.strip()
        return text

    async def _clean_text_for_chunking(self, text: str) -> str:
        """Làm sạch text"""
        if not text:
            return text

        # Bước 1: Bảo vệ số quan trọng
        protected_text, placeholders = self._protect_important_numbers(text)

        # Bước 2: Xóa reference numbers
        cleaned_text = self._remove_reference_numbers(protected_text)

        # Bước 3: Chuẩn hóa whitespace và newlines
        normalized_text = self._normalize_whitespace_and_newlines(cleaned_text)

        # Bước 4: Khôi phục số đã bảo vệ
        final_text = self._restore_protected_numbers(normalized_text, placeholders)

        # Bước 5: Làm sạch khoảng trắng thừa
        final_text = re.sub(r"\s+", " ", final_text).strip()

        if self.debug_mode:
            logger.debug("Text after cleaning: %s", final_text)

        return final_text

    def _classify_block_type(self, text: str, spans: List[dict]) -> str:
        text_stripped = text.strip()
        if not text_stripped:
            return "paragraph"

        if re.search(r"\|[-|]+\|", text_stripped) or (
            len(text_stripped) < 50
            and re.match(r"^\d+[\.\,]\d+|\d+\s*[-|]\s*\d+", text_stripped)
        ):
            return "table"

        for pattern in self.HEADING_PATTERNS:
            if re.match(pattern, text_stripped, re.IGNORECASE):
                return "title"

        for pattern in self.LIST_PATTERNS:
            if re.match(pattern, text_stripped):
                return "list"

        for pattern in self.CODE_PATTERNS:
            if re.match(pattern, text_stripped, re.IGNORECASE):
                return "title"

        if spans and len(spans) > 0:
            avg_font_size = sum(span["size"] for span in spans) / len(spans)
            if avg_font_size > 14:
                return "title"

        return "paragraph"

    def _calculate_merged_bbox(self, blocks: List[TextBlock]) -> Dict:
        """Tính toán bbox sau khi merge các blocks"""
        if not blocks:
            return {"left": 0, "top": 0, "right": 0, "bottom": 0}

        left = min(block.metadata.bbox["left"] for block in blocks)
        top = min(block.metadata.bbox["top"] for block in blocks)
        right = max(block.metadata.bbox["right"] for block in blocks)
        bottom = max(block.metadata.bbox["bottom"] for block in blocks)
        return {"left": left, "top": top, "right": right, "bottom": bottom}

    async def _merge_title_with_paragraph(
        self, blocks: List[TextBlock]
    ) -> List[TextBlock]:
        """Merge title với paragraph liền kề - CHỈ merge, KHÔNG tokenize"""
        merged_blocks = []
        i = 0

        while i < len(blocks):
            current_block = blocks[i]

            if (
                current_block.metadata.block_type == "title"
                and i + 1 < len(blocks)
                and blocks[i + 1].metadata.block_type == "paragraph"
            ):
                next_block = blocks[i + 1]

                if (
                    abs(
                        current_block.metadata.bbox["bottom"]
                        - next_block.metadata.bbox["top"]
                    )
                    < 20
                ):
                    merged_text = (
                        f"{current_block.context.strip()} {next_block.context.strip()}"
                    )

                    merged_text = await self._clean_text_for_chunking(merged_text)

                    merged_metadata = BlockMetadata(
                        bbox=self._calculate_merged_bbox([current_block, next_block]),
                        block_type="title_with_content",
                        num_lines=current_block.metadata.num_lines
                        + next_block.metadata.num_lines,
                        num_spans=current_block.metadata.num_spans
                        + next_block.metadata.num_spans,
                        is_cleaned=current_block.metadata.is_cleaned,
                        page_index=current_block.metadata.page_index,
                        language_info=current_block.metadata.language_info,
                    )

                    merged_block = TextBlock(
                        block_id=f"{current_block.block_id}_merged",
                        context=merged_text,
                        metadata=merged_metadata,
                    )
                    merged_blocks.append(merged_block)
                    i += 2
                else:
                    merged_blocks.append(current_block)
                    i += 1
            else:
                merged_blocks.append(current_block)
                i += 1

        return merged_blocks

    async def _merge_short_blocks(self, blocks: List[TextBlock]) -> List[TextBlock]:
        """Gộp các block ngắn bất kể loại block - CHỈ merge, KHÔNG tokenize"""
        merged_blocks = []
        i = 0

        while i < len(blocks):
            current_block = blocks[i]

            if len(current_block.context) < self.max_block_length:
                short_blocks = [current_block]
                j = i + 1

                while (
                    j < len(blocks)
                    and len(blocks[j].context) < self.max_block_length
                    and abs(
                        blocks[j].metadata.bbox["top"]
                        - short_blocks[-1].metadata.bbox["bottom"]
                    )
                    < self.max_bbox_distance
                ):
                    short_blocks.append(blocks[j])
                    j += 1

                if len(short_blocks) > 1:
                    # CHỈ merge, KHÔNG tokenize
                    merged_text = " ".join(
                        block.context.strip() for block in short_blocks
                    )

                    # Chỉ clean basic
                    merged_text = await self._clean_text_for_chunking(merged_text)

                    # Giữ block_type của block đầu tiên hoặc đặt là 'mixed' nếu các block có loại khác nhau
                    block_types = {block.metadata.block_type for block in short_blocks}
                    merged_block_type = short_blocks[0].metadata.block_type if len(block_types) == 1 else "mixed"

                    merged_metadata = BlockMetadata(
                        bbox=self._calculate_merged_bbox(short_blocks),
                        block_type=merged_block_type,
                        num_lines=sum(
                            block.metadata.num_lines for block in short_blocks
                        ),
                        num_spans=sum(
                            block.metadata.num_spans for block in short_blocks
                        ),
                        is_cleaned=short_blocks[0].metadata.is_cleaned,
                        page_index=short_blocks[0].metadata.page_index,
                        language_info=short_blocks[0].metadata.language_info,
                    )

                    merged_block = TextBlock(
                        block_id=f"{short_blocks[0].block_id}_merged",
                        context=merged_text,
                        metadata=merged_metadata,
                    )
                    merged_blocks.append(merged_block)
                    i = j
                else:
                    merged_blocks.append(current_block)
                    i += 1
            else:
                merged_blocks.append(current_block)
                i += 1

        return merged_blocks

    async def _merge_table_blocks(self, blocks: List[TextBlock]) -> List[TextBlock]:
        """Merge các table blocks liên tiếp - CHỈ merge, KHÔNG tokenize"""
        merged_blocks = []
        i = 0

        while i < len(blocks):
            current_block = blocks[i]

            if current_block.metadata.block_type == "table":
                table_blocks = [current_block]
                j = i + 1

                while j < len(blocks) and blocks[j].metadata.block_type == "table":
                    if (
                        abs(
                            blocks[j].metadata.bbox["top"]
                            - table_blocks[-1].metadata.bbox["bottom"]
                        )
                        < 20
                    ):
                        table_blocks.append(blocks[j])
                        j += 1
                    else:
                        break

                if len(table_blocks) > 1:
                    # CHỈ merge, KHÔNG tokenize
                    merged_text = " ".join(
                        block.context.strip() for block in table_blocks
                    )

                    # Chỉ clean basic
                    merged_text = await self._clean_text_for_chunking(merged_text)

                    merged_metadata = BlockMetadata(
                        bbox=self._calculate_merged_bbox(table_blocks),
                        block_type="table",
                        num_lines=sum(
                            block.metadata.num_lines for block in table_blocks
                        ),
                        num_spans=sum(
                            block.metadata.num_spans for block in table_blocks
                        ),
                        is_cleaned=table_blocks[0].metadata.is_cleaned,
                        page_index=table_blocks[0].metadata.page_index,
                        language_info=table_blocks[0].metadata.language_info,
                    )

                    merged_block = TextBlock(
                        block_id=f"table_{table_blocks[0].block_id}",
                        context=merged_text,
                        metadata=merged_metadata,
                    )
                    merged_blocks.append(merged_block)
                    i = j
                else:
                    merged_blocks.append(current_block)
                    i += 1
            else:
                merged_blocks.append(current_block)
                i += 1

        return merged_blocks

    async def _apply_final_stopword_removal(
        self, blocks: List[TextBlock]
    ) -> List[TextBlock]:
        """Apply stopword removal cuối cùng, sau khi merge xong"""
        if not (self.remove_stopwords and self.tokenizer and self.text_detector):
            return blocks

        processed_blocks = []
        for block in blocks:
            try:
                # Detect language từ context (text đẹp)
                language_info = self.text_detector.detect_language(block.context)

                # Remove stopwords từ context (for chunking) nhưng preserve format
                processed_context = await self.tokenizer.tokenize_preserve_format(
                    block.context, language_info
                )

                # Tạo block mới với context đã remove stopwords
                processed_block = TextBlock(
                    block_id=block.block_id,
                    context=processed_context,  # Đã remove stopwords nhưng vẫn đẹp
                    metadata=block.metadata,
                )
                processed_blocks.append(processed_block)

            except Exception as e:
                logger.warning(
                    f"Lỗi khi remove stopwords cho block {block.block_id}: {e}"
                )
                # Fallback: giữ nguyên block gốc
                processed_blocks.append(block)

        return processed_blocks

    async def extract_text_blocks_from_page(self, page) -> List[TextBlock]:
        """Trích xuất text blocks từ một page"""
        try:
            text_dict = page.get_text("dict")
            blocks = []

            for block_idx, block in enumerate(text_dict["blocks"]):
                if "lines" not in block:
                    continue

                block_text = ""
                all_spans = []

                for line in block["lines"]:
                    line_text = ""
                    for span in line["spans"]:
                        span_text = span["text"]
                        line_text += span_text
                        all_spans.append(span)

                    if line_text.strip():
                        block_text += line_text + "\n"

                if block_text.strip():
                    original_text = block_text.strip()

                    if self.debug_mode:
                        logger.debug(
                            "Processing block %s: %s", block_idx, original_text
                        )

                    if TextFilter.is_likely_metadata(original_text):
                        if self.debug_mode:
                            logger.debug(
                                "Block %s filtered out (metadata): %s",
                                block_idx,
                                original_text,
                            )
                        continue

                    block_type = self._classify_block_type(original_text, all_spans)

                    if self.enable_text_cleaning:
                        cleaned_text = TextFilter.clean_text(
                            original_text,
                            remove_urls=self.remove_urls,
                            remove_page_numbers=self.remove_page_numbers,
                            remove_short_lines=self.remove_short_lines,
                            min_line_length=self.min_line_length,
                        )

                        if not cleaned_text:
                            if self.debug_mode:
                                logger.debug(
                                    "Block %s filtered out (empty after cleaning): %s",
                                    block_idx,
                                    original_text,
                                )
                            continue

                        cleaned_text = await self._clean_text_for_chunking(cleaned_text)
                    else:
                        cleaned_text = await self._clean_text_for_chunking(
                            original_text
                        )

                    if not cleaned_text.strip():
                        continue
                    if self.debug_mode:
                        logger.debug("Original: %s", repr(original_text))
                        logger.debug("Cleaned: %s", repr(cleaned_text))

                    language_info = None
                    if self.text_detector:
                        try:
                            language_info = self.text_detector.detect_language(
                                cleaned_text
                            )
                        except Exception as e:
                            logger.error("Lỗi khi phát hiện ngôn ngữ: %s", e)

                    bbox_coords = block["bbox"]
                    bbox = BBox(
                        left=bbox_coords[0],
                        top=bbox_coords[1],
                        right=bbox_coords[2],
                        bottom=bbox_coords[3],
                    )

                    metadata = BlockMetadata(
                        bbox=bbox.to_dict(),
                        block_type=block_type,
                        num_lines=len(block["lines"]),
                        num_spans=len(all_spans),
                        is_cleaned=self.enable_text_cleaning,
                        page_index=0,
                        language_info=language_info,
                    )

                    text_block = TextBlock(
                        block_id=block_idx,
                        context=cleaned_text,
                        metadata=metadata,
                    )
                    blocks.append(text_block)

            # Merge các blocks
            blocks = await self._merge_title_with_paragraph(blocks)
            blocks = await self._merge_short_blocks(blocks)
            blocks = await self._merge_table_blocks(blocks)

            # Apply stopword removal cuối cùng
            blocks = await self._apply_final_stopword_removal(blocks)

            return blocks

        except Exception as e:
            logger.error("Lỗi khi trích xuất blocks từ page: %s", e)
            return []

    async def extract_all_pages_data(self, pdf_path: str) -> List[PageData]:
        """Trích xuất data từ tất cả các pages"""
        try:
            doc = fitz.open(pdf_path)
            all_pages = []
            total_pages = len(doc)

            logger.info("Bắt đầu xử lý file PDF: %s (%d trang)", pdf_path, total_pages)

            for page_num in range(total_pages):
                page = doc[page_num]
                blocks = await self.extract_text_blocks_from_page(page)

                for block in blocks:
                    block.metadata.page_index = page_num

                page_data = PageData(
                    page_index=page_num,
                    page_size=PageSize(width=page.rect.width, height=page.rect.height),
                    blocks=blocks,
                )
                all_pages.append(page_data)

                if self.debug_mode:
                    logger.debug("Page %d: %d blocks", page_num, len(blocks))

            doc.close()
            total_blocks = sum(len(page.blocks) for page in all_pages)
            logger.info(
                "Đã xử lý xong %d trang, tổng số blocks: %d", total_pages, total_blocks
            )
            return all_pages

        except FileNotFoundError:
            logger.error("Không tìm thấy file PDF: %s", pdf_path)
            raise
        except Exception as e:
            logger.error("Lỗi khi xử lý PDF: %s", e)
            raise

    async def extract_all_blocks_to_json(self, pdf_path: str, output_path: str) -> None:
        """Trích xuất tất cả blocks và lưu vào JSON"""
        from dataclasses import asdict

        try:
            pages_data = await self.extract_all_pages_data(pdf_path)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump([asdict(page) for page in pages_data], f, ensure_ascii=False, indent=2)

            logger.info("Đã lưu kết quả vào: %s", output_path)

        except Exception as e:
            logger.error("Lỗi khi lưu file JSON: %s", e)
            raise

async def main():
    """Main function để test"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    pdf_path = "C:/Users/Quangdepzai/Downloads/text.pdf"
    output_path = "output_blocks.json"

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
        debug_mode=True,
    )

    try:
        await extractor.extract_all_blocks_to_json(pdf_path, output_path)
    except FileNotFoundError:
        logger.error("Không tìm thấy file PDF. Vui lòng kiểm tra lại đường dẫn.")
        raise
    except ImportError as e:
        logger.error("Thiếu thư viện: %s", e)
        raise
    except Exception as e:
        logger.error("Lỗi không xác định: %s", e)
        raise

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())