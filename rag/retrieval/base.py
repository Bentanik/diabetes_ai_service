from typing import List, Dict
from abc import ABC, abstractmethod

class BaseRetriever(ABC):
    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        pass