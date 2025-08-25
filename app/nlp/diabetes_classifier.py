import asyncio
from concurrent.futures import ThreadPoolExecutor
from transformers import pipeline
import torch
from rag.dataclasses import DocumentChunk
from typing import List
import gc
import logging

logger = logging.getLogger(__name__)

class DiabetesClassifier:
    _instance = None

    def __new__(cls, model_name: str = "joeddav/xlm-roberta-large-xnli"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_model(model_name)
        return cls._instance

    def _init_model(self, model_name: str):
        self.device = 0 if torch.cuda.is_available() else -1
        self.executor = ThreadPoolExecutor(max_workers=1)
        
        try:
            self.classifier = pipeline(
                "zero-shot-classification",
                model=model_name,
                device=self.device,
                torch_dtype=torch.float16 if self.device >= 0 else torch.float32,
                truncation=True  # Quan trọng: tránh lỗi với text dài
            )
            logger.info(f"Model loaded on {'GPU' if self.device >= 0 else 'CPU'}")
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise

    def _has_negation(self, text: str, keyword: str) -> bool:
        """Kiểm tra phủ định trong phạm vi gần keyword"""
        text_lower = text.lower()
        idx = text_lower.find(keyword)
        if idx == -1:
            return False
        start = max(0, idx - 100)
        context = text_lower[start:idx]
        negations = ['not', 'no', 'non-', 'without', 'never', 'rarely', 'không', 'chưa', 'chẳng']
        return any(neg in context for neg in negations)

    def _score_batch_sync(self, texts: List[str]) -> List[float]:
        if not texts:
            return []

        try:
            with torch.no_grad():
                scores = []
                for text in texts:
                    if not text.strip():
                        scores.append(0.0)
                        continue

                    text_lower = text.lower()

                    # === 1. Zero-shot classification ===
                    try:
                        result = self.classifier(
                            text,
                            candidate_labels=[
                                "diabetes, insulin, HbA1c, diabetic complications",
                                "other diseases and general health",
                                "lifestyle, nutrition, exercise"
                            ],
                            hypothesis_template="The text is about {}."
                        )
                        diabetes_score = result['scores'][0]
                        other_medical_score = result['scores'][1]
                        lifestyle_score = result['scores'][2]
                    except Exception as e:
                        logger.warning(f"Zero-shot failed for text: {e}")
                        diabetes_score = 0.0
                        other_medical_score = 0.0
                        lifestyle_score = 0.0

                    # === 2. Keyword matching (cả tiếng Việt + Anh) ===
                    diabetes_keywords = [
                        # English
                        'diabetes', 'diabetic', 'insulin', 'glucose', 'blood sugar',
                        'hyperglycemia', 'hypoglycemia', 'type 1', 'type 2',
                        'ketoacidosis', 'hba1c', 'glycemic', 'metformin',
                        'glycosylated hemoglobin', 'pancreas', 'islet cells',
                        'neuropathy', 'retinopathy', 'nephropathy',
                        # Vietnamese
                        'tiểu đường', 'đái tháo đường', 'tăng đường huyết',
                        'hba1c', 'insulin', 'metformin', 'biến chứng tiểu đường',
                        'suy thận', 'tổn thương thần kinh', 'bệnh võng mạc'
                    ]

                    keyword_count = 0
                    keyword_weight = 0.0
                    for kw in diabetes_keywords:
                        if kw in text_lower:
                            if not self._has_negation(text_lower, kw):
                                keyword_count += 1
                                # Tăng trọng số cho từ chuyên sâu
                                if kw in ['hba1c', 'insulin', 'metformin', 'ketoacidosis', 'HbA1c']:
                                    keyword_weight += 0.3
                                elif kw in ['type 1', 'type 2', 'neuropathy', 'retinopathy']:
                                    keyword_weight += 0.2
                                else:
                                    keyword_weight += 0.15

                    # === 3. Logic scoring — ưu tiên keyword nếu model không chắc ===
                    if lifestyle_score > 0.6:
                        final_score = 0.0  # Rõ ràng không liên quan
                    elif diabetes_score < 0.3 and keyword_count == 0:
                        final_score = 0.0
                    elif keyword_count >= 3:
                        # Nhiều keyword → chắc chắn liên quan
                        final_score = min(0.4 + keyword_weight, 1.0)
                    elif keyword_count >= 2 and diabetes_score > 0.25:
                        final_score = min(diabetes_score + 0.3 + keyword_weight, 0.9)
                    elif keyword_count >= 1 and diabetes_score > 0.4:
                        final_score = min(diabetes_score + 0.25 + keyword_weight, 0.8)
                    else:
                        # Dù model score thấp, nhưng có keyword → không loại hoàn toàn
                        final_score = max(diabetes_score * 0.7, keyword_weight * 0.8)

                    scores.append(round(final_score, 3))

                return scores

        except torch.cuda.OutOfMemoryError:
            logger.warning("OOM: splitting batch")
            torch.cuda.empty_cache()
            if len(texts) > 1:
                mid = len(texts) // 2
                left = self._score_batch_sync(texts[:mid])
                right = self._score_batch_sync(texts[mid:])
                return left + right
            return [0.0]
        except Exception as e:
            logger.error(f"Scoring error: {e}", exc_info=True)
            return [0.0] * len(texts)
        finally:
            if self.device >= 0:
                torch.cuda.empty_cache()
            gc.collect()

    async def score_chunk(self, chunk_text: str) -> float:
        if not chunk_text or not chunk_text.strip():
            return 0.0
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                self.executor, self._score_batch_sync, [chunk_text]
            )
            return result[0] if result else 0.0
        except Exception as e:
            logger.warning(f"Score chunk failed: {e}")
            return 0.0

    async def score_chunks(self, chunks: List[DocumentChunk], batch_size: int = 4) -> List[float]:
        if not chunks:
            return []

        all_scores = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c.content for c in batch if c.content.strip()]

            if texts:
                loop = asyncio.get_running_loop()
                try:
                    batch_scores = await loop.run_in_executor(
                        self.executor, self._score_batch_sync, texts
                    )
                    all_scores.extend(batch_scores)
                except Exception as e:
                    logger.error(f"Batch scoring failed: {e}")
                    all_scores.extend([0.0] * len(texts))
            else:
                all_scores.extend([0.0] * len(batch))

        return all_scores

    def cleanup(self):
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        if self.device >= 0:
            torch.cuda.empty_cache()