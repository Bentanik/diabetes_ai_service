from pydantic import BaseModel
from typing import List
from app.database.models import SettingModel

class SettingModelDTO(BaseModel):
    list_knowledge_ids: List[str]
    top_k: int
    search_accuracy: float
    temperature: float

    @classmethod
    def from_model(cls, model: SettingModel):
        return cls(
            list_knowledge_ids=model.list_knowledge_ids,
            top_k=model.top_k,
            search_accuracy=model.search_accuracy,
            temperature=model.temperature,
        )