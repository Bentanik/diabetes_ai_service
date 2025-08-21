import asyncio
from concurrent.futures import ThreadPoolExecutor
from transformers import XLMRobertaTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from rag.dataclasses import DocumentChunk
from typing import List
import gc

executor = ThreadPoolExecutor(max_workers=1)

class DiabetesClassifier:
    _instance = None

    def __new__(cls, model_name: str = "joeddav/xlm-roberta-large-xnli"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_model(model_name)
        return cls._instance

    def _init_model(self, model_name: str):
        # Chọn device
        self.device = 0 if torch.cuda.is_available() else -1
        print(f"Đang tải mô hình zero-shot đa ngôn ngữ trên {'GPU' if self.device == 0 else 'CPU'}...")

        # FP16 nếu GPU
        dtype = torch.float16 if self.device == 0 else torch.float32

        # Load tokenizer và model
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, torch_dtype=dtype
        ).to(self.device if self.device >= 0 else "cpu")

        # Pipeline
        self.classifier = pipeline(
            "zero-shot-classification",
            model=self.model,
            tokenizer=self.tokenizer
        )
        print("Tải mô hình thành công.")

    # Hàm scoring synchronous (chạy trong executor)
    def _score_chunk_sync(self, chunk_text: str) -> float:
        labels = ["diabetes", "tiểu đường", "not diabetes", "không tiểu đường"]
        with torch.no_grad():
            result = self.classifier(chunk_text, candidate_labels=labels)
        diabetes_score = sum(
            result['scores'][result['labels'].index(label)]
            for label in ["diabetes", "tiểu đường"] if label in result['labels']
        )
        # clear cache tạm
        torch.cuda.empty_cache()
        gc.collect()
        return min(diabetes_score * 1.5, 1.0)

    # Async wrapper
    async def score_chunk(self, chunk_text: str) -> float:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, self._score_chunk_sync, chunk_text)

    # Async batch scoring
    async def score_chunks(self, chunks: List[DocumentChunk], batch_size: int = 1) -> List[float]:
        scores = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            texts = [c.content for c in batch]
            tasks = [self.score_chunk(t) for t in texts]
            batch_scores = await asyncio.gather(*tasks)
            scores.extend(batch_scores)
        return scores
