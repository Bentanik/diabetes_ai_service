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
                torch_dtype=torch.float16 if self.device >= 0 else torch.float32
            )
            print(f"Model loaded on {'GPU' if self.device >= 0 else 'CPU'}")
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise

    def _score_batch_sync(self, texts: List[str]) -> List[float]:
        if not texts:
            return []
            
        try:
            with torch.no_grad():
                scores = []
                
                for text in texts:
                    # Step 1: Check if it's specifically about diabetes
                    diabetes_result = self.classifier(
                        text,
                        candidate_labels=[
                            "diabetes, blood sugar, insulin, diabetic complications",
                            "other medical conditions and general health topics",
                            "non-medical content"
                        ]
                    )
                    
                    diabetes_score = diabetes_result['scores'][0]
                    medical_score = diabetes_result['scores'][1]
                    non_medical_score = diabetes_result['scores'][2]
                    
                    diabetes_keywords = [
                        'diabetes', 'diabetic', 'insulin', 'glucose', 'blood sugar',
                        'hyperglycemia', 'hypoglycemia', 'type 1', 'type 2',
                        'diabetic ketoacidosis', 'hba1c', 'glycemic', 'metformin',
                        'glycosylated hemoglobin', 'pancreas', 'islet cells',
                        'diabetic neuropathy', 'diabetic retinopathy', 'diabetic nephropathy'
                    ]
                    
                    text_lower = text.lower()
                    keyword_count = sum(1 for keyword in diabetes_keywords if keyword in text_lower)
                    
                    # Step 3: Use all 3 scores for better logic
                    # If non-medical dominates, reject immediately
                    if non_medical_score > 0.6:
                        final_score = 0.0
                    
                    # If medical but not diabetes-specific enough
                    elif medical_score > diabetes_score + 0.15 and keyword_count == 0:
                        final_score = 0.0
                    
                    # Strong diabetes signals
                    elif diabetes_score > 0.35 and keyword_count > 0 and diabetes_score > non_medical_score:
                        # Strong diabetes signal + keywords → boost to 80-95%
                        base_boost = 0.4
                        keyword_boost = min(keyword_count * 0.15, 0.35)
                        # Extra boost if diabetes clearly beats non-medical
                        dominance_boost = 0.1 if diabetes_score > non_medical_score + 0.2 else 0.0
                        final_score = min(diabetes_score + base_boost + keyword_boost + dominance_boost, 0.95)
                        
                    elif diabetes_score > 0.45 and diabetes_score > medical_score + 0.1 and diabetes_score > non_medical_score:
                        # Diabetes clearly dominates both medical and non-medical
                        dominance_boost = 0.15 if diabetes_score > max(medical_score, non_medical_score) + 0.2 else 0.1
                        final_score = min(diabetes_score + 0.3 + dominance_boost, 0.9)
                        
                    elif keyword_count >= 3 and diabetes_score > 0.25 and diabetes_score >= non_medical_score:
                        # Many diabetes keywords + not non-medical
                        final_score = min(diabetes_score + 0.45, 0.85)
                        
                    elif keyword_count >= 2 and diabetes_score > 0.3 and diabetes_score > non_medical_score + 0.1:
                        # Multiple diabetes keywords + clearly medical
                        final_score = min(diabetes_score + 0.35, 0.8)
                        
                    elif keyword_count >= 1 and diabetes_score > 0.4 and diabetes_score > non_medical_score:
                        # Some keywords + good diabetes score + not non-medical
                        final_score = min(diabetes_score + 0.25, 0.75)
                        
                    else:
                        # Not diabetes-specific enough or too non-medical
                        final_score = 0.0
                    
                    scores.append(final_score)
                
                return scores
                
        except torch.cuda.OutOfMemoryError:
            logger.warning("OOM detected, splitting batch")
            torch.cuda.empty_cache()
            
            if len(texts) > 1:
                mid = len(texts) // 2
                return (self._score_batch_sync(texts[:mid]) + 
                    self._score_batch_sync(texts[mid:]))
            return [0.0]
            
        except Exception as e:
            logger.error(f"Scoring error: {e}")
            return [0.0] * len(texts)
        finally:
            if self.device >= 0:
                torch.cuda.empty_cache()
            gc.collect()

    async def score_chunk(self, chunk_text: str) -> float:
        if not chunk_text.strip():
            return 0.0
        loop = asyncio.get_running_loop()
        try:
            scores = await loop.run_in_executor(
                self.executor, self._score_batch_sync, [chunk_text]
            )
            return scores[0] if scores else 0.0
        except Exception:
            return 0.0

    async def score_chunks(self, chunks: List[DocumentChunk], batch_size: int = 2) -> List[float]:
        if not chunks:
            return []
            
        all_scores = []
        # Giảm batch size vì giờ xử lý phức tạp hơn
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
                except Exception:
                    all_scores.extend([0.0] * len(texts))
            else:
                all_scores.extend([0.0] * len(batch))
                
        return all_scores

    def cleanup(self):
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        if self.device >= 0:
            torch.cuda.empty_cache()
