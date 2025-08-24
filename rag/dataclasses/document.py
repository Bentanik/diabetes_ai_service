from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class ParsedContent:
    content: str
    metadata: Dict[str, Any]
    file_type: str
    file_path: str
    tables: List[List[List[str]]] = None

    def __post_init__(self):
        if self.tables is None:
            self.tables = []