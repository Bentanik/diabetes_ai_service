from pydantic import BaseModel, Field
from typing import Generic, TypeVar, List

T = TypeVar("T")


class Pagination(BaseModel, Generic[T]):
    items: List[T] = Field(default_factory=list)
    total: int
    page: int
    limit: int

    @property
    def total_pages(self) -> int:
        return (self.total + self.limit - 1) // self.limit

    class Config:
        arbitrary_types_allowed = True
