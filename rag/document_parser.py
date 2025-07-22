import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import re

from utils.logger_utils import get_logger
from utils.vietnamese_language_utils import VietnameseLanguageUtils

try:
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
except ImportError:
    from langchain.document_loaders import PyPDFLoader, TextLoader

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

try:
    import fitz  # PyMuPDF để tách bố cục PDF

    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import pdfplumber  # Tách bảng PDF

    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import pandas as pd  # Định dạng bảng đẹp hơn

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from docx import Document as DocxDocument  # Load thư viện DOCX

    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

logger = get_logger(__name__)


class DocumentParser:
    def __init__(self, vi_words_json: Path):
        self.supported_formats = {".pdf", ".txt", ".md", ".csv"}
        if DOCX_AVAILABLE:
            self.supported_formats.add(".docx")
        self._vi_words_json = vi_words_json
        self.vn_utils = VietnameseLanguageUtils(self._vi_words_json)
        self.capabilities = self._check_capabilities()
        self._log_capabilities()

    def _check_capabilities(self) -> Dict[str, bool]:
        return {
            "pymupdf": PYMUPDF_AVAILABLE,
            "pdfplumber": PDFPLUMBER_AVAILABLE,
            "pandas": PANDAS_AVAILABLE,
            "docx": DOCX_AVAILABLE,
        }

    def _log_capabilities(self):
        if PYMUPDF_AVAILABLE:
            logger.info("Có PyMuPDF (phân tích PDF nâng cao)")
        else:
            logger.warning("Thiếu PyMuPDF - sẽ dùng PyPDFLoader cơ bản")
        if PDFPLUMBER_AVAILABLE:
            logger.info("Có pdfplumber (tách bảng PDF)")
        else:
            logger.warning("Thiếu pdfplumber - không tách được bảng PDF")
        if PANDAS_AVAILABLE:
            logger.info("Có pandas (xử lý bảng)")
        else:
            logger.warning("Thiếu pandas - hạn chế xử lý bảng")
        if DOCX_AVAILABLE:
            logger.info("Có python-docx (hỗ trợ DOCX)")
        else:
            logger.warning("Thiếu python-docx - không đọc được file Word")

    def is_noise_line(self, text: str, y0: Optional[float] = None) -> bool:
        text = text.strip().lower()
        if re.match(r"^\d+\s+of\s+\d+$", text):
            return True
        if re.match(r"^trang\s*\d+", text):
            return True
        if re.match(r"^page\s*\d+", text):
            return True
        if re.match(r"^(http|www)[\w\.\-\/]+", text):
            return True
        if re.match(r"^\d{1,2}:\d{2}", text):
            return True
        if re.match(r"^\d+$", text):
            return True
        if re.match(r"\d{1,2}/\d{1,2}/\d{2,4}", text):
            return True
        if y0 is not None and (y0 < 20 or y0 > 800):
            return True
        return False

    def detect_content_type(self, text: str) -> str:
        if self._looks_like_table(text):
            return "table"
        if self._looks_like_image_content(text):
            return "image"
        if self._looks_like_header(text):
            return "header"
        if self._looks_like_list(text):
            return "list"
        return "text"

    def _looks_like_table(self, text: str) -> bool:
        lines = text.split("\n")
        if len(lines) < 2:
            return False
        sep = sum(1 for l in lines if re.search(r"\s{2,}|\t|[|]", l))
        return sep >= len(lines) * 0.5

    def _looks_like_image_content(self, text: str) -> bool:
        image_keywords = [
            "hình",
            "ảnh",
            "biểu đồ",
            "chart",
            "figure",
            "graph",
            "diagram",
        ]
        return any(word in text.lower() for word in image_keywords)

    def _looks_like_header(self, text: str) -> bool:
        if re.search(r"(^|\n)\s*(-|\d+\.|[a-zA-Z]\.)", text):
            return False
        if len(text.split()) <= 15 and (text.isupper() or text.istitle()):
            return True
        return False

    def _looks_like_list(self, text: str) -> bool:
        lines = text.split("\n")
        list_lines = sum(
            1 for l in lines if re.match(r"^(-|\d+\.|[a-zA-Z]\.)", l.strip())
        )
        return list_lines >= len(lines) * 0.5

    def extract_metadata(self, text: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        language = self.vn_utils.detect_language(text)
        return {
            **meta,
            "length": len(text),
            "words": len(text.split()),
            "content_type": self.detect_content_type(text),
            "language": language,
            "detected": language,
            "score": self.vn_utils._calculate_vietnamese_score(text),
        }

    def load_pdf(self, path: str) -> List[Document]:
        docs = []
        if not PYMUPDF_AVAILABLE or not PDFPLUMBER_AVAILABLE:
            logger.warning("Thiếu PyMuPDF hoặc pdfplumber, dùng PyPDFLoader fallback")
            loader = PyPDFLoader(path)
            return loader.load()

        with pdfplumber.open(path) as pdf:
            doc = fitz.open(path)
            for page_num, page in enumerate(pdf.pages):
                blocks = []
                tables = page.find_tables()
                table_regions = []
                for table_num, tb in enumerate(tables or []):
                    bx0, by0, bx1, by1 = tb.bbox
                    table_regions.append((bx0 - 5, by0 - 5, bx1 + 5, by1 + 5))
                    rows = ["\t".join([c or "" for c in r]) for r in tb.extract()]
                    text = "\n".join(rows)
                    cleaned = self.vn_utils.clean_vietnamese_text(text)
                    if cleaned:
                        meta = self.extract_metadata(
                            cleaned,
                            {
                                "type": "table",
                                "page": page_num,
                                "table_num": table_num,
                                "source": path,
                                "extraction_method": "pdfplumber",
                                "x0": bx0,
                                "y0": by0,
                                "x1": bx1,
                                "y1": by1,
                            },
                        )
                        blocks.append(
                            {
                                "y0": by0,
                                "x0": bx0,
                                "type": "table",
                                "page_content": cleaned,
                                "metadata": meta,
                            }
                        )

                pymupdf_page = doc.load_page(page_num)
                text_blocks = sorted(
                    pymupdf_page.get_textpage().extractBLOCKS(),
                    key=lambda b: (round(b[1], 1), round(b[0], 1)),
                )
                current_block = []
                current_coords = None

                for b in text_blocks:
                    x0, y0, x1, y1, text, *_ = b
                    if not text.strip() or self.is_noise_line(text, y0):
                        continue
                    is_in_table = False
                    for bx0, by0, bx1, by1 in table_regions:
                        if x0 >= bx0 and x1 <= bx1 and y0 >= by0 and y1 <= by1:
                            is_in_table = True
                            break
                    if is_in_table:
                        continue

                    cleaned = self.vn_utils.clean_vietnamese_text(text)
                    if not cleaned:
                        continue

                    if current_block and (
                        current_coords and y0 - current_coords[3] < 20
                    ):
                        current_block.append(cleaned)
                        if current_coords:
                            current_coords = (
                                min(current_coords[0], x0),
                                current_coords[1],
                                max(current_coords[2], x1),
                                max(current_coords[3], y1),
                            )
                    else:
                        if current_block:
                            full_text = "\n".join(current_block)
                            meta = self.extract_metadata(
                                full_text,
                                {
                                    "type": "text",
                                    "page": page_num,
                                    "source": path,
                                    "extraction_method": "pymupdf_blocks",
                                    "x0": current_coords[0] if current_coords else x0,
                                    "y0": current_coords[1] if current_coords else y0,
                                    "x1": current_coords[2] if current_coords else x1,
                                    "y1": current_coords[3] if current_coords else y1,
                                },
                            )
                            blocks.append(
                                {
                                    "y0": current_coords[1] if current_coords else y0,
                                    "x0": current_coords[0] if current_coords else x0,
                                    "type": "text",
                                    "page_content": full_text,
                                    "metadata": meta,
                                }
                            )
                        current_block = [cleaned]
                        current_coords = (x0, y0, x1, y1)

                if current_block:
                    full_text = "\n".join(current_block)
                    meta = self.extract_metadata(
                        full_text,
                        {
                            "type": "text",
                            "page": page_num,
                            "x0": current_coords[0] if current_coords else x0,
                            "y0": current_coords[1] if current_coords else y0,
                            "x1": current_coords[2] if current_coords else x1,
                            "y1": current_coords[3] if current_coords else y1,
                        },
                    )
                    blocks.append(
                        {
                            "y0": current_coords[1] if current_coords else y0,
                            "x0": current_coords[0] if current_coords else x0,
                            "type": "text",
                            "page_content": full_text,
                            "metadata": meta,
                        }
                    )

                blocks = sorted(
                    blocks, key=lambda b: (round(b["y0"], 1), round(b["x0"], 1))
                )
                for blk in blocks:
                    docs.append(
                        Document(
                            page_content=blk["page_content"], metadata=blk["metadata"]
                        )
                    )
            doc.close()
        return docs

    def load_document(self, path: str) -> List[Document]:
        if not os.path.exists(path):
            logger.error(f"Không tìm thấy file: {path}")
            return []
        ext = Path(path).suffix.lower()
        if ext == ".pdf":
            return self.load_pdf(path)
        else:
            logger.error(f"Không hỗ trợ định dạng: {ext}")
            logger.info(
                f"Định dạng được hỗ trợ: {', '.join(sorted(self.supported_formats))}"
            )
            return []


def main():
    parser = DocumentParser(Path("shared/vietnamese_words.json"))
    docs = parser.load_document("test_document.pdf")

    # Lưu ra file JSON
    output_data = [
        {"content": doc.page_content, "metadata": doc.metadata} for doc in docs
    ]
    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print("Đã lưu kết quả ra file output.json")


if __name__ == "__main__":
    main()
