from typing import Any, List, Dict
from abc import ABC, abstractmethod

class BaseRetriever(ABC):
    @abstractmethod
    async def retrieve(
        self,
        query_vector: List[float],
        query_text: str = None,
        score_threshold: float = 0.0,
        **filters
    ) -> List[Dict[str, Any]]:
        """
        Tìm kiếm tài liệu liên quan
        """
        pass