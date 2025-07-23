import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import re
import fitz  # PyMuPDF
import pdfplumber
from langchain.schema import Document
from utils.logger_utils import get_logger
from utils.vietnamese_language_utils import VietnameseLanguageUtils

logger = get_logger(__name__)


class DocumentParser:
    def __init__(self, vi_words_json: Path):
        self.vn_utils = VietnameseLanguageUtils(vi_words_json)
        self.noise_patterns = [
            r"^\d+\s+of\s+\d+$",
            r"^trang\s*\d+",
            r"^page\s*\d+",
            r"^(http|www)[\w\.\-\/]+",
            r"^\d{1,2}:\d{2}",
            r"^\d+$",
            r"\d{1,2}/\d{1,2}/\d{2,4}",
        ]

    def is_noise_line(self, text: str, y0: Optional[float] = None) -> bool:
        text = text.strip().lower()
        if len(text) < 3:  # Quá ngắn
            return True
        if any(re.match(pat, text) for pat in self.noise_patterns):
            return True
        if y0 is not None and (y0 < 30 or y0 > 800):  # Header/footer area
            return True
        return False

    def clean_text(self, text: str) -> str:
        return self.vn_utils.clean_vietnamese_text(text)

    def classify_text_block(self, text: str) -> str:
        text = text.strip()
        if not text:
            return "empty"

        lines = text.split("\n")

        # Heading: ít từ, chữ viết hoa hoặc chữ đầu mỗi từ viết hoa (Title Case)
        if len(lines) == 1 and len(text.split()) <= 12:
            if text.isupper() or text.istitle() or re.match(r"^[A-Z\d\s\.\-]+$", text):
                return "heading"

        # List: có dòng bắt đầu bằng - hoặc số + dấu chấm hoặc chữ + dấu chấm
        list_lines = sum(
            1
            for line in lines
            if line.strip()
            and re.match(r"^(\s*[-•]\s*|\s*\d+[\.\)]\s*|\s*[a-zA-Z][\.\)]\s*)", line)
        )
        if list_lines >= len(lines) * 0.6 and list_lines >= 2:
            return "list"

        # Caption: đoạn text chứa từ khóa chỉ dẫn hình ảnh, biểu đồ
        image_keywords = [
            "hình",
            "ảnh",
            "biểu đồ",
            "chart",
            "figure",
            "graph",
            "diagram",
            "bảng",
            "table",
            "nguồn:",
            "source:",
            "ghi chú:",
            "note:",
        ]
        if any(word in text.lower() for word in image_keywords) and len(lines) <= 3:
            return "caption"

        # Mặc định coi là đoạn văn bản bình thường
        return "paragraph"

    def extract_metadata(self, text: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        lang = self.vn_utils.detect_language(text)
        return {
            **meta,
            "length": len(text),
            "words": len(text.split()),
            "language": lang,
            "score": self.vn_utils._calculate_vietnamese_score(text),
        }

    def format_table_for_rag(
        self, table_data: List[List[str]], table_num: int, page_num: int
    ) -> str:
        """
        Format table data for RAG processing
        Returns a structured text representation of the table
        """
        if not table_data or not table_data[0]:
            return ""

        # Lọc bỏ các row/cell trống
        filtered_data = []
        for row in table_data:
            filtered_row = [cell.strip() if cell else "" for cell in row]
            if any(cell for cell in filtered_row):  # Có ít nhất 1 cell không rỗng
                filtered_data.append(filtered_row)

        if not filtered_data:
            return ""

        # Format cho RAG: dạng structured text
        headers = filtered_data[0] if filtered_data else []
        rows = filtered_data[1:] if len(filtered_data) > 1 else []

        # Tạo text representation cho RAG
        rag_text = f"Bảng {table_num + 1} (Trang {page_num + 1}):\n"

        # Nếu có headers
        if headers and any(h.strip() for h in headers):
            rag_text += "Tiêu đề cột: " + " | ".join(h.strip() for h in headers) + "\n"

        # Thêm các dòng dữ liệu
        for i, row in enumerate(rows):
            if any(cell.strip() for cell in row):
                if headers and len(headers) == len(row):
                    # Format với tên cột
                    row_data = []
                    for j, cell in enumerate(row):
                        if cell.strip() and j < len(headers) and headers[j].strip():
                            row_data.append(f"{headers[j].strip()}: {cell.strip()}")
                    if row_data:
                        rag_text += f"Dòng {i + 1}: " + "; ".join(row_data) + "\n"
                else:
                    # Format đơn giản
                    row_cells = [cell.strip() for cell in row if cell.strip()]
                    if row_cells:
                        rag_text += f"Dòng {i + 1}: " + " | ".join(row_cells) + "\n"

        return rag_text.strip()

    def get_table_json_for_frontend(
        self, table_data: List[List[str]]
    ) -> Dict[str, Any]:
        """
        Format table data for frontend display
        """
        if not table_data:
            return {"headers": [], "rows": []}

        # Lọc và làm sạch dữ liệu
        filtered_data = []
        for row in table_data:
            filtered_row = [cell.strip() if cell else "" for cell in row]
            if any(cell for cell in filtered_row):
                filtered_data.append(filtered_row)

        if not filtered_data:
            return {"headers": [], "rows": []}

        # Xác định headers và rows
        headers = filtered_data[0] if filtered_data else []
        rows = filtered_data[1:] if len(filtered_data) > 1 else []

        # Đảm bảo consistency về số cột
        max_cols = max(len(row) for row in filtered_data) if filtered_data else 0

        # Chuẩn hóa headers
        normalized_headers = []
        for i in range(max_cols):
            if i < len(headers) and headers[i].strip():
                normalized_headers.append(headers[i].strip())
            else:
                normalized_headers.append(f"Cột {i + 1}")

        # Chuẩn hóa rows
        normalized_rows = []
        for row in rows:
            normalized_row = []
            for i in range(max_cols):
                if i < len(row):
                    normalized_row.append(row[i].strip())
                else:
                    normalized_row.append("")
            normalized_rows.append(normalized_row)

        return {
            "headers": normalized_headers,
            "rows": normalized_rows,
            "total_rows": len(normalized_rows),
            "total_cols": max_cols,
        }

    def should_merge_blocks(
        self, block1: Dict, block2: Dict, threshold: float = 25.0
    ) -> bool:
        """
        Xác định có nên merge 2 text blocks hay không
        """
        if not block1 or not block2:
            return False

        bbox1 = block1.get("bbox", (0, 0, 0, 0))
        bbox2 = block2.get("bbox", (0, 0, 0, 0))

        # Kiểm tra khoảng cách vertical
        vertical_gap = bbox2[1] - bbox1[3]  # y0_block2 - y1_block1

        # Kiểm tra overlap horizontal
        horizontal_overlap = min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0])
        width_avg = (bbox1[2] - bbox1[0] + bbox2[2] - bbox2[0]) / 2

        # Merge nếu:
        # 1. Khoảng cách vertical nhỏ
        # 2. Có overlap horizontal đáng kể
        # 3. Cùng loại block (heading với heading, paragraph với paragraph)
        return (
            vertical_gap < threshold
            and horizontal_overlap > width_avg * 0.3
            and block1.get("type") == block2.get("type")
        )

    def load_pdf(self, path: str) -> List[Document]:
        docs = []
        try:
            pdf = pdfplumber.open(path)
            doc = fitz.open(path)
        except Exception as e:
            logger.error(f"Lỗi mở file PDF: {e}")
            return []

        for page_num, page in enumerate(pdf.pages):
            table_bboxes = []

            # Xử lý tables trước
            for table_num, table in enumerate(page.find_tables() or []):
                bx0, by0, bx1, by1 = table.bbox
                table_bboxes.append((bx0, by0, bx1, by1))

                table_data = table.extract()
                if not table_data:
                    continue

                # Format cho RAG
                rag_text = self.format_table_for_rag(table_data, table_num, page_num)
                if not rag_text:
                    continue

                # Format cho Frontend
                frontend_data = self.get_table_json_for_frontend(table_data)

                meta = self.extract_metadata(
                    rag_text,
                    {
                        "type": "table",
                        "page": page_num,
                        "table_num": table_num,
                        "bbox": (bx0, by0, bx1, by1),
                        "source": path,
                        "table_data": frontend_data,  # Dữ liệu cho frontend
                        "raw_table": table_data,  # Dữ liệu thô nếu cần
                    },
                )

                docs.append(Document(page_content=rag_text, metadata=meta))

            # Xử lý text blocks
            pymupdf_page = doc.load_page(page_num)
            text_blocks = sorted(
                pymupdf_page.get_textpage().extractBLOCKS(),
                key=lambda b: (round(b[1], 1), round(b[0], 1)),
            )

            processed_blocks = []

            for b in text_blocks:
                x0, y0, x1, y1, text, *_ = b
                if not text.strip() or self.is_noise_line(text, y0):
                    continue

                # Loại bỏ text nằm trong bảng
                in_table = any(
                    x0 >= bx0 - 5 and x1 <= bx1 + 5 and y0 >= by0 - 5 and y1 <= by1 + 5
                    for (bx0, by0, bx1, by1) in table_bboxes
                )
                if in_table:
                    continue

                cleaned = self.clean_text(text)
                if not cleaned:
                    continue

                block_info = {
                    "text": cleaned,
                    "bbox": (x0, y0, x1, y1),
                    "type": self.classify_text_block(cleaned),
                }

                processed_blocks.append(block_info)

            # Merge các blocks liền kề
            merged_blocks = []
            i = 0
            while i < len(processed_blocks):
                current_block = processed_blocks[i]
                merged_text = current_block["text"]
                merged_bbox = current_block["bbox"]
                block_type = current_block["type"]

                # Tìm các blocks tiếp theo có thể merge
                j = i + 1
                while j < len(processed_blocks):
                    next_block = processed_blocks[j]
                    if self.should_merge_blocks(
                        {"bbox": merged_bbox, "type": block_type},
                        {"bbox": next_block["bbox"], "type": next_block["type"]},
                    ):
                        merged_text += "\n" + next_block["text"]
                        merged_bbox = (
                            min(merged_bbox[0], next_block["bbox"][0]),
                            merged_bbox[1],
                            max(merged_bbox[2], next_block["bbox"][2]),
                            max(merged_bbox[3], next_block["bbox"][3]),
                        )
                        j += 1
                    else:
                        break

                # Re-classify sau khi merge
                final_type = self.classify_text_block(merged_text)

                meta = self.extract_metadata(
                    merged_text,
                    {
                        "type": final_type,
                        "page": page_num,
                        "bbox": merged_bbox,
                        "source": path,
                    },
                )

                docs.append(Document(page_content=merged_text, metadata=meta))
                i = j

        doc.close()
        pdf.close()

        # Lọc bỏ các documents quá ngắn hoặc không có ý nghĩa
        filtered_docs = []
        for doc in docs:
            content = doc.page_content.strip()
            if (
                len(content) >= 10 and len(content.split()) >= 3
            ):  # Ít nhất 10 ký tự và 3 từ
                filtered_docs.append(doc)

        return filtered_docs

    def load_document(self, path: str) -> List[Document]:
        if not os.path.exists(path):
            logger.error(f"Không tìm thấy file: {path}")
            return []
        ext = Path(path).suffix.lower()
        if ext == ".pdf":
            return self.load_pdf(path)
        else:
            logger.error(f"Không hỗ trợ định dạng: {ext}")
            return []


if __name__ == "__main__":
    parser = DocumentParser(Path("shared/vietnamese_words.json"))
    docs = parser.load_document("C:/Users/vietv/Downloads/download.pdf")
    print(f"Parsed {len(docs)} documents")

    output_data = []
    for doc in docs:
        doc_data = {"content": doc.page_content, "metadata": doc.metadata}

        # Đặc biệt format cho table
        if doc.metadata.get("type") == "table":
            doc_data["table_data"] = doc.metadata.get("table_data", {})

        output_data.append(doc_data)

    import json

    with open("parsed_output.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print("Đã lưu kết quả ra file parsed_output.json")
