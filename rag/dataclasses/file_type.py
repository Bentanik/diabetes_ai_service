from enum import Enum

class FileType(Enum):
    PDF = "PDF"
    DOCX = "DOCX" 
    TXT = "Plain Text"
    HTML = "HTML"
    MARKDOWN = "Markdown"
    CSV = "CSV"
    EXCEL = "Excel"
    
    def __str__(self) -> str:
        """Return human-readable string"""
        return self.value
    
    @property
    def extensions(self) -> list[str]:
        """Return common extensions for this file type"""
        mapping = {
            FileType.PDF: ['.pdf'],
            FileType.DOCX: ['.docx', '.doc'],
            FileType.TXT: ['.txt'],
            FileType.HTML: ['.html', '.htm'],
            FileType.MARKDOWN: ['.md', '.markdown'],
            FileType.CSV: ['.csv'],
            FileType.EXCEL: ['.xlsx', '.xls']
        }
        return mapping.get(self, [])
