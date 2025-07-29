"""
PDF Text Extractor - Trích xuất text blocks từ PDF

Module này cung cấp khả năng trích xuất và xử lý text blocks từ file PDF:
- Sử dụng PyMuPDF (fitz) để extract text với độ chính xác cao
- Hỗ trợ làm sạch text tự động (loại bỏ URL, số trang, noise)
- Xuất kết quả ra format JSON với metadata chi tiết
- Tối ưu cho tiếng Việt và các ngôn ngữ Unicode
"""

import json
import logging
from typing import List

from rag.document_parser.text_filter import TextFilter
from rag.document_parser.models.pdf_types import (
    BBox,
    TextBlock,
    BlockMetadata,
    PageData,
    PageSize,
)

try:
    import fitz

    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    raise ImportError("PyMuPDF là bắt buộc. Cài đặt với: pip install PyMuPDF")

logger = logging.getLogger(__name__)


class PdfExtractor:
    """
    PDF Text Extractor sử dụng PyMuPDF

    Trích xuất text blocks từ PDF với khả năng làm sạch text tự động.
    Tối ưu cho tiếng Việt và hỗ trợ Unicode đầy đủ.
    """

    def __init__(
        self,
        enable_text_cleaning: bool = True,
        remove_urls: bool = True,
        remove_page_numbers: bool = True,
        remove_short_lines: bool = True,
        min_line_length: int = 3,
    ):
        """
        Khởi tạo PDF Extractor

        Args:
            enable_text_cleaning: Bật/tắt làm sạch text
            remove_urls: Loại bỏ URLs và email
            remove_page_numbers: Loại bỏ số trang (6/15, Page 1, etc.)
            remove_short_lines: Loại bỏ dòng quá ngắn
            min_line_length: Độ dài tối thiểu của dòng
        """
        self.enable_text_cleaning = enable_text_cleaning
        self.remove_urls = remove_urls
        self.remove_page_numbers = remove_page_numbers
        self.remove_short_lines = remove_short_lines
        self.min_line_length = min_line_length

        if not PYMUPDF_AVAILABLE:
            raise ImportError("PyMuPDF is required but not installed")

    def extract_text_blocks_from_page(self, page) -> List[TextBlock]:
        """
        Trích xuất text blocks từ một trang PDF

        Args:
            page: PyMuPDF page object

        Returns:
            List[TextBlock]: Danh sách các text blocks với typed metadata
        """
        try:
            text_dict = page.get_text("dict")
            blocks = []

            for block_idx, block in enumerate(text_dict["blocks"]):
                if "lines" not in block:
                    continue

                # Gộp text từ tất cả lines trong block
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

                    # Làm sạch text nếu được bật
                    if self.enable_text_cleaning:
                        cleaned_text = TextFilter.clean_text(
                            original_text,
                            remove_urls=self.remove_urls,
                            remove_page_numbers=self.remove_page_numbers,
                            remove_short_lines=self.remove_short_lines,
                            min_line_length=self.min_line_length,
                        )

                        if not cleaned_text or TextFilter.is_likely_metadata(
                            original_text
                        ):
                            continue

                        final_text = cleaned_text
                    else:
                        final_text = original_text

                    if not final_text.strip():
                        continue

                    # Tính toán bounding box
                    bbox_coords = block["bbox"]
                    bbox = BBox(
                        left=bbox_coords[0],
                        top=bbox_coords[1],
                        right=bbox_coords[2],
                        bottom=bbox_coords[3],
                    )

                    # Tạo metadata với proper dataclass
                    metadata = BlockMetadata(
                        bbox=bbox.to_dict(),
                        block_type="paragraph",
                        num_lines=len(block["lines"]),
                        num_spans=len(all_spans),
                        is_cleaned=self.enable_text_cleaning,
                    )

                    # Tạo TextBlock với proper dataclass
                    text_block = TextBlock(
                        block_id=block_idx,
                        context=final_text,
                        metadata=metadata,
                    )
                    blocks.append(text_block)

            return blocks

        except Exception as e:
            logger.error("Lỗi khi trích xuất blocks: %s", e)
            return []

    def extract_all_pages_data(self, pdf_path: str) -> List[PageData]:
        """
        Trích xuất data từ tất cả các trang PDF

        Args:
            pdf_path: Đường dẫn file PDF input

        Returns:
            List[PageData]: Danh sách data của tất cả các trang

        Raises:
            FileNotFoundError: Nếu không tìm thấy file PDF
            Exception: Lỗi khi xử lý PDF
        """
        try:
            doc = fitz.open(pdf_path)
            all_pages = []
            total_pages = len(doc)

            logger.info("Bắt đầu xử lý file PDF: %s (%d trang)", pdf_path, total_pages)

            for page_num in range(total_pages):
                page = doc[page_num]
                blocks = self.extract_text_blocks_from_page(page)

                for block in blocks:
                    block.metadata.page_index = page_num

                page_data = PageData(
                    page_index=page_num,
                    page_size=PageSize(width=page.rect.width, height=page.rect.height),
                    blocks=blocks,
                )

                all_pages.append(page_data)

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

    def extract_all_blocks_to_json(self, pdf_path: str, output_path: str) -> None:
        """
        Trích xuất blocks từ tất cả các trang và lưu vào JSON

        Args:
            pdf_path: Đường dẫn file PDF input
            output_path: Đường dẫn file JSON output

        Raises:
            FileNotFoundError: Nếu không tìm thấy file PDF
            Exception: Lỗi khi xử lý PDF
        """
        try:
            pages_data = self.extract_all_pages_data(pdf_path)

            json_data = []
            for page in pages_data:
                page_dict = {
                    "page_index": page.page_index,
                    "page_size": {
                        "width": page.page_size.width,
                        "height": page.page_size.height,
                    },
                    "blocks": [
                        {
                            "block_id": block.block_id,
                            "context": block.context,
                            "metadata": {
                                "bbox": block.metadata.bbox,
                                "block_type": block.metadata.block_type,
                                "num_lines": block.metadata.num_lines,
                                "num_spans": block.metadata.num_spans,
                                "is_cleaned": block.metadata.is_cleaned,
                                "page_index": block.metadata.page_index,
                            },
                        }
                        for block in page.blocks
                    ],
                }
                json_data.append(page_dict)

            # Lưu vào JSON
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)

            logger.info("Đã lưu kết quả vào: %s", output_path)

        except Exception as e:
            logger.error("Lỗi khi lưu file JSON: %s", e)
            raise


def main():
    """Demo function - ví dụ sử dụng PdfExtractor"""
    # Configure logging for demo
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    pdf_path = "C:/Users/vietv/Downloads/asd.pdf"
    output_path = "output_blocks_clean.json"

    # Tạo extractor với text cleaning enabled
    extractor = PdfExtractor(
        enable_text_cleaning=True,
        remove_urls=True,
        remove_page_numbers=True,
        remove_short_lines=True,
        min_line_length=3,
    )

    try:
        extractor.extract_all_blocks_to_json(pdf_path, output_path)
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
    main()
