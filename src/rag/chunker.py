"""
Bộ xử lý tài liệu PDF.
Hỗ trợ trích xuất văn bản và chunking.
"""

import os
import re
from typing import List, Dict, Any, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredFileLoader,
)
from langchain.schema import Document
from core.logging_config import get_logger

logger = get_logger(__name__)


class Chunker:
    """
    Bộ phân tích PDF.
    Cải thiện việc trích xuất văn bản với nhận diện layout.
    """

    def __init__(self):
        self.outlines = []

    def __call__(
        self, filename: str, from_page: int = 0, to_page: int = 100000, **kwargs
    ) -> List[Dict]:
        """
        Phân tích file PDF và trích xuất nội dung có cấu trúc.

        Args:
            filename: Đường dẫn file PDF
            from_page: Trang bắt đầu (0-indexed)
            to_page: Trang kết thúc

        Returns:
            Danh sách dictionary chứa nội dung từng trang
        """
        try:
            # Thử sử dụng pypdf để trích xuất text cơ bản
            return self._parse_with_pypdf(filename, from_page, to_page)
        except Exception as e:
            logger.warning(
                f"PyPDF parsing thất bại: {e}, chuyển sang phương pháp đơn giản"
            )
            return self._simple_text_extraction(filename, from_page, to_page)

    def _parse_with_pypdf(
        self, filename: str, from_page: int, to_page: int
    ) -> List[Dict]:
        """Phân tích sử dụng PyPDF với trích xuất text tốt hơn."""
        from pypdf import PdfReader

        pages = []
        reader = PdfReader(filename)
        total_pages = len(reader.pages)

        for page_num in range(from_page, min(to_page, total_pages)):
            try:
                page = reader.pages[page_num]
                text = page.extract_text()

                # Làm sạch và cấu trúc hóa text
                text = self._clean_vietnamese_text(text)

                if text.strip():
                    pages.append(
                        {
                            "page_number": page_num,
                            "content": text,
                            "type": "text",
                            "bbox": None,  # Cần phân tích layout phức tạp
                            "metadata": {
                                "source": filename,
                                "page": page_num,
                                "extraction_method": "pypdf_ragflow",
                            },
                        }
                    )

            except Exception as e:
                logger.warning(f"Lỗi xử lý trang {page_num}: {e}")
                continue

        return pages

    def _simple_text_extraction(
        self, filename: str, from_page: int, to_page: int
    ) -> List[Dict]:
        """Phương pháp trích xuất text đơn giản dự phòng."""
        try:
            loader = PyPDFLoader(filename)
            documents = loader.load()

            pages = []
            for i, doc in enumerate(documents[from_page:to_page]):
                if doc.page_content.strip():
                    pages.append(
                        {
                            "page_number": from_page + i,
                            "content": self._clean_vietnamese_text(doc.page_content),
                            "type": "text",
                            "bbox": None,
                            "metadata": {
                                "source": filename,
                                "page": from_page + i,
                                "extraction_method": "langchain_pypdf",
                            },
                        }
                    )
            return pages

        except Exception as e:
            logger.error(f"Tất cả phương pháp phân tích PDF đều thất bại: {e}")
            return []

    def _clean_vietnamese_text(self, text: str) -> str:
        """Làm sạch text được trích xuất - tối ưu cho tiếng Việt."""
        if not text:
            return ""

        # Loại bỏ khoảng trắng thừa
        text = re.sub(r"\s+", " ", text)

        # Loại bỏ patterns header/footer trang
        text = re.sub(r"^\d+\s*$", "", text, flags=re.MULTILINE)

        # Sửa lỗi trích xuất PDF thông thường
        text = re.sub(
            r"([a-zà-ỹ])([A-ZÀ-Ỹ])", r"\1 \2", text
        )  # Thêm space giữa chữ thường và hoa
        text = re.sub(r"([.!?])([A-ZÀ-Ỹ])", r"\1 \2", text)  # Thêm space sau dấu câu

        # Tối ưu cho tiếng Việt
        text = re.sub(r"([ăâêôơưĂÂÊÔƠƯ])([A-ZĂÂÊÔƠƯ])", r"\1 \2", text)

        # Sửa lỗi ngắt dòng giữa từ
        text = re.sub(r"([a-zà-ỹ])-\s*\n([a-zà-ỹ])", r"\1\2", text)

        return text.strip()


class VietnameseDocumentChunker:
    """Bộ chia nhỏ tài liệu và tối ưu tiếng Việt."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        use_ragflow_pdf: bool = True,
    ):
        """
        Khởi tạo bộ chia nhỏ tài liệu.

        Args:
            chunk_size: Kích thước chunk tối đa
            chunk_overlap: Độ chồng lấp giữa các chunks
            separators: Các ký tự phân tách tùy chỉnh
            use_ragflow_pdf: Sử dụng RAGFlow PDF parser
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_ragflow_pdf = use_ragflow_pdf

        # Separators tối ưu cho tiếng Việt và cấu trúc tài liệu
        self.separators = separators or [
            "\n\n",  # Ngắt đoạn văn
            "\n",  # Ngắt dòng
            ". ",  # Kết thúc câu
            "! ",  # Cảm thán
            "? ",  # Nghi vấn
            "。",  # Dấu chấm
            "！",  # Cảm thán
            "？",  # Nghi vấn
            "; ",  # Chấm phẩy
            ": ",  # Hai chấm
            " ",  # Khoảng trắng
            "",  # Cấp độ ký tự
        ]

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False,
        )

        # Khởi tạo RAGFlow PDF parser
        if self.use_ragflow_pdf:
            self.pdf_parser = Chunker()

        logger.info(
            f"Đã khởi tạo VietnameseDocumentChunker: chunk_size={chunk_size}, "
            f"overlap={chunk_overlap}, ragflow_pdf={use_ragflow_pdf}"
        )

    def load_file(self, file_path: str) -> List[Document]:
        """Tải file với parsing cải tiến."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File không tồn tại: {file_path}")

        file_ext = os.path.splitext(file_path)[1].lower()

        try:
            if file_ext == ".pdf" and self.use_ragflow_pdf:
                return self._load_pdf_with_ragflow(file_path)
            elif file_ext == ".pdf":
                loader = PyPDFLoader(file_path)
            elif file_ext in [".docx", ".doc"]:
                loader = Docx2txtLoader(file_path)
            elif file_ext == ".txt":
                loader = TextLoader(file_path, encoding="utf-8")
            else:
                loader = UnstructuredFileLoader(file_path)

            documents = loader.load()
            logger.info(f"Đã tải {len(documents)} documents từ {file_path}")
            return documents

        except Exception as e:
            logger.error(f"Lỗi tải file {file_path}: {e}")
            raise

    def _load_pdf_with_ragflow(self, file_path: str) -> List[Document]:
        """Tải PDF sử dụng RAGFlow parser."""
        try:
            pages_data = self.pdf_parser(file_path)
            documents = []

            for page_data in pages_data:
                doc = Document(
                    page_content=page_data["content"],
                    metadata={
                        "source": file_path,
                        "page": page_data["page_number"],
                        "type": page_data["type"],
                        "extraction_method": page_data["metadata"]["extraction_method"],
                        **page_data["metadata"],
                    },
                )
                documents.append(doc)

            logger.info(
                f"RAGFlow PDF parser đã trích xuất {len(documents)} trang từ {file_path}"
            )
            return documents

        except Exception as e:
            logger.error(f"RAGFlow PDF parsing thất bại: {e}, chuyển sang loader chuẩn")
            loader = PyPDFLoader(file_path)
            return loader.load()

    def chunk_documents(
        self, documents: List[Document], preserve_structure: bool = True
    ) -> List[Document]:
        """
        Chia nhỏ documents với bảo toàn cấu trúc.

        Args:
            documents: Danh sách documents đầu vào
            preserve_structure: Có bảo toàn cấu trúc tài liệu không

        Returns:
            Danh sách documents đã được chia nhỏ
        """
        try:
            if preserve_structure:
                chunks = self._chunk_with_structure_preservation(documents)
            else:
                chunks = self.text_splitter.split_documents(documents)

            # Metadata bổ sung
            for i, chunk in enumerate(chunks):
                chunk.metadata.update(
                    {
                        "chunk_id": i,
                        "chunk_size": len(chunk.page_content),
                        "total_chunks": len(chunks),
                        "chunk_method": (
                            "structure_aware" if preserve_structure else "standard"
                        ),
                        "vietnamese_optimized": True,
                    }
                )

            logger.info(f"Đã chia {len(documents)} documents → {len(chunks)} chunks")
            return chunks

        except Exception as e:
            logger.error(f"Lỗi chia nhỏ documents: {e}")
            raise

    def _chunk_with_structure_preservation(
        self, documents: List[Document]
    ) -> List[Document]:
        """Chia nhỏ với bảo toàn cấu trúc tốt hơn."""
        all_chunks = []

        for doc in documents:
            # Nhận diện cấu trúc tài liệu
            content = doc.page_content

            # Chia theo các thành phần cấu trúc chính trước
            major_sections = self._split_by_vietnamese_structure(content)

            # Sau đó áp dụng recursive splitting cho từng section
            for section in major_sections:
                if len(section.strip()) > self.chunk_size:
                    # Section lớn - cần chia nhỏ tiếp
                    section_doc = Document(
                        page_content=section, metadata=doc.metadata.copy()
                    )
                    section_chunks = self.text_splitter.split_documents([section_doc])
                    all_chunks.extend(section_chunks)
                else:
                    # Section nhỏ - giữ nguyên
                    chunk = Document(page_content=section, metadata=doc.metadata.copy())
                    all_chunks.append(chunk)

        return all_chunks

    def _split_by_vietnamese_structure(self, content: str) -> List[str]:
        """Chia nội dung theo các thành phần cấu trúc tiếng Việt."""
        # Tìm các patterns cấu trúc tài liệu phổ biến
        patterns = [
            r"\n\n(?=[A-ZÀ-Ỹ])",  # Đoạn văn mới bắt đầu bằng chữ hoa
            r"\n(?=\d+\.)",  # Danh sách có số
            r"\n(?=[A-ZÀ-Ỹ][a-zà-ỹ]+:)",  # Tiêu đề section có dấu hai chấm
            r"\n(?=•)",  # Bullet points
            r"\n(?=-)",  # Dash points
            r"\n(?=Chương \d+)",  # Chương
            r"\n(?=Phần \d+)",  # Phần
            r"\n(?=Mục \d+)",  # Mục
        ]

        sections = [content]

        for pattern in patterns:
            new_sections = []
            for section in sections:
                parts = re.split(pattern, section)
                new_sections.extend([p for p in parts if p.strip()])
            sections = new_sections

        return sections

    def process_file(
        self, file_path: str, preserve_structure: bool = True
    ) -> List[Document]:
        """Xử lý file với parsing và chunking cải tiến."""
        documents = self.load_file(file_path)
        return self.chunk_documents(documents, preserve_structure)

    def process_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        preserve_structure: bool = True,
    ) -> List[Document]:
        """Xử lý raw text thành chunks."""
        doc = Document(page_content=text, metadata=metadata or {})
        return self.chunk_documents([doc], preserve_structure)

    def process_multiple_files(
        self, file_paths: List[str], preserve_structure: bool = True
    ) -> List[Document]:
        """Xử lý nhiều files với parsing cải tiến."""
        all_chunks = []

        for file_path in file_paths:
            try:
                chunks = self.process_file(file_path, preserve_structure)

                # Thêm metadata nguồn
                for chunk in chunks:
                    chunk.metadata.update(
                        {
                            "source_file": os.path.basename(file_path),
                            "source_path": file_path,
                        }
                    )

                all_chunks.extend(chunks)

            except Exception as e:
                logger.error(f"Lỗi xử lý file {file_path}: {e}")
                continue

        logger.info(f"Đã xử lý {len(file_paths)} files → {len(all_chunks)} chunks")
        return all_chunks


# Global instance
_vietnamese_chunker = None


def get_vietnamese_chunker(
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    use_ragflow_pdf: bool = True,
) -> VietnameseDocumentChunker:
    """Lấy instance chunker tối ưu cho tiếng Việt."""
    global _vietnamese_chunker
    if _vietnamese_chunker is None:
        _vietnamese_chunker = VietnameseDocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            use_ragflow_pdf=use_ragflow_pdf,
        )
    return _vietnamese_chunker
