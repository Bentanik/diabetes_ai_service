from typing import TypeVar
from .result import Result

T = TypeVar("T")

# Type alias cho None result
ResultVoid = Result[None]
