# diabetes_scorer.py
from transformers import XLMRobertaTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from rag.dataclasses import DocumentChunk
from typing import List

class DiabetesClassifier:
    _instance = None

    def __new__(cls, model_name: str = "joeddav/xlm-roberta-large-xnli"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_model(model_name)
        return cls._instance

    def _init_model(self, model_name: str):
        self.device = 0 if torch.cuda.is_available() else -1
        print(f"Đang tải mô hình zero-shot đa ngôn ngữ trên {'GPU' if self.device == 0 else 'CPU'}...")

        self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.classifier = pipeline(
            "zero-shot-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device
        )
        print("Tải mô hình thành công.")

    def score_chunk(self, chunk_text: str) -> float:
        labels = ["diabetes", "tiểu đường", "not diabetes", "không tiểu đường"]
        result = self.classifier(chunk_text, candidate_labels=labels)
        diabetes_score = sum(
            result['scores'][result['labels'].index(label)]
            for label in ["diabetes", "tiểu đường"] if label in result['labels']
        )
        return min(diabetes_score * 1.5, 1.0)

    def score_chunks(self, chunks: List[DocumentChunk], batch_size: int = 8) -> List[float]:
        scores = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            texts = [c.content for c in batch]
            results = self.classifier(texts, candidate_labels=["diabetes", "tiểu đường", "not diabetes", "không tiểu đường"])
            if isinstance(results, dict):
                results = [results]
            for result in results:
                diabetes_score = sum(
                    result['scores'][result['labels'].index(label)]
                    for label in ["diabetes", "tiểu đường"] if label in result['labels']
                )
                scores.append(min(diabetes_score * 1.5, 1.0))
        return scores
