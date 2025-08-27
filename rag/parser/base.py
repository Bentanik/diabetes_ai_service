from pathlib import Path
from typing import List, Union

class BaseParser:
    def _validate_file(self, file_path: Union[str, Path]) -> Path:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return path

    def get_file_extensions(self) -> List[str]:
        raise NotImplementedError

    def get_file_type(self) -> str:
        raise NotImplementedError

    async def parse_async(self, file_path: Union[str, Path]):
        raise NotImplementedError