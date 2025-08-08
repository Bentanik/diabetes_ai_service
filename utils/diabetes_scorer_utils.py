import json
import os
import re
import time
import asyncio
import pickle
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain.schema import Document
from utils import get_logger

logger = get_logger(__name__)


class DiabetesAnalysisResult:
    def __init__(
        self,
        final_score: float,
        semantic_score: float,
        keyword_score: float,
        relevance_level: str,
        word_count: int,
    ):
        self.final_score = final_score
        self.semantic_score = semantic_score
        self.keyword_score = keyword_score
        self.relevance_level = relevance_level
        self.word_count = word_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_score": self.final_score,
            "semantic_score": self.semantic_score,
            "keyword_score": self.keyword_score,
            "relevance_level": self.relevance_level,
            "word_count": self.word_count,
        }

    def __str__(self) -> str:
        return (
            f"DiabetesAnalysis(score={self.final_score}, level={self.relevance_level})"
        )

    def __repr__(self) -> str:
        return self.__str__()


class DiabetesScorer:
    def __init__(self, data_dir: str = "shared", model_dir: str = "model"):
        self.data_dir = Path.cwd() / data_dir
        self.model_dir = Path.cwd() / model_dir
        self.keywords_file = self.data_dir / "diabetes_words.json"
        self.embeddings_file = self.model_dir / "diabetes_embeddings.pkl"
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        self.model = None
        self.keywords = None
        self.embeddings = None
        self._initialized = False
        self._initializing = False
        self._init_lock = asyncio.Lock()
        self.high_threshold = 0.6
        self.medium_threshold = 0.35

    async def _ensure_initialized(self):
        from core.embedding import EmbeddingModel

        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            if self._initializing:
                while self._initializing and not self._initialized:
                    await asyncio.sleep(0.1)
                return
            self._initializing = True
            try:
                logger.info("Khởi tạo DiabetesScorer...")
                start_time = time.time()
                embedding_model = await EmbeddingModel.get_instance()
                self.model = embedding_model.model
                self.keywords = await asyncio.to_thread(self._load_keywords)
                self.embeddings = await self._load_embeddings()
                self._initialized = True
                logger.info(
                    f"DiabetesScorer đã sẵn sàng trong {time.time() - start_time:.2f}s"
                )
            except Exception as e:
                logger.error(f"Lỗi khởi tạo DiabetesScorer: {str(e)}", exc_info=True)
                raise
            finally:
                self._initializing = False

    def _load_keywords(self) -> Dict[str, Any]:
        if self.keywords_file.exists():
            try:
                with open(self.keywords_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(
                    f"Lỗi load keywords từ {self.keywords_file}: {str(e)}",
                    exc_info=True,
                )
        keywords = {
            "vietnamese": {
                "primary": [
                    "tiểu đường",
                    "đái tháo đường",
                    "bệnh tiểu đường",
                    "tiểu đường type 1",
                    "tiểu đường type 2",
                    "đường huyết",
                    "glucose",
                    "insulin",
                    "HbA1c",
                ],
                "medical": [
                    "nội tiết",
                    "tuyến tụy",
                    "hormone insulin",
                    "kháng insulin",
                    "tăng đường huyết",
                    "hạ đường huyết",
                ],
                "symptoms": ["khát nước nhiều", "đi tiểu nhiều", "mệt mỏi", "sụt cân"],
                "treatments": ["metformin", "insulin therapy", "thuốc hạ đường huyết"],
            },
            "english": {
                "primary": [
                    "diabetes",
                    "diabetes mellitus",
                    "type 1 diabetes",
                    "type 2 diabetes",
                    "blood glucose",
                    "insulin",
                    "HbA1c",
                ],
                "medical": [
                    "endocrine",
                    "pancreas",
                    "insulin resistance",
                    "hyperglycemia",
                    "hypoglycemia",
                ],
                "symptoms": [
                    "excessive thirst",
                    "frequent urination",
                    "fatigue",
                    "weight loss",
                ],
                "treatments": [
                    "metformin",
                    "insulin therapy",
                    "antidiabetic medication",
                ],
            },
            "weights": {
                "primary": 2.0,
                "medical": 1.5,
                "treatments": 1.8,
                "symptoms": 1.2,
            },
        }
        try:
            with open(self.keywords_file, "w", encoding="utf-8") as f:
                json.dump(keywords, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(
                f"Lỗi lưu keywords vào {self.keywords_file}: {str(e)}", exc_info=True
            )
        return keywords

    async def _load_embeddings(self) -> Dict[str, Any]:
        if self.embeddings_file.exists():
            try:
                data = await asyncio.to_thread(self._load_pickle_file)
                if data and "embeddings" in data and len(data["embeddings"]) > 0:
                    logger.info(f"Loaded {len(data['texts'])} embeddings từ cache")
                    return data
            except Exception as e:
                logger.error(
                    f"Lỗi load cache từ {self.embeddings_file}: {str(e)}", exc_info=True
                )
        return await self._create_embeddings()

    def _load_pickle_file(self) -> Dict[str, Any]:
        with open(self.embeddings_file, "rb") as f:
            return pickle.load(f)

    async def _create_embeddings(self) -> Dict[str, Any]:
        logger.info("Đang tạo embeddings...")
        start_time = time.time()
        all_texts = []
        for lang in ["vietnamese", "english"]:
            for category, words in self.keywords[lang].items():
                all_texts.extend([w for w in words if isinstance(w, str)])
        unique_texts = list(set(all_texts))
        logger.info(f"Tạo embeddings cho {len(unique_texts)} từ khóa...")
        embeddings = await asyncio.to_thread(self.model.embed_documents, unique_texts)
        category_embeddings = {}
        for lang in ["vietnamese", "english"]:
            for category, words in self.keywords[lang].items():
                if words:
                    cat_embeddings = await asyncio.to_thread(
                        self.model.embed_documents, words
                    )
                    category_embeddings[f"{lang}_{category}"] = np.mean(
                        cat_embeddings, axis=0
                    )
        result = {
            "texts": unique_texts,
            "embeddings": embeddings,
            "mean_embedding": np.mean(embeddings, axis=0),
            "category_embeddings": category_embeddings,
        }
        try:
            await asyncio.to_thread(self._save_pickle_file, result)
            logger.info(
                f"Embeddings đã tạo và cache trong {time.time() - start_time:.2f}s"
            )
        except Exception as e:
            logger.error(f"Không thể lưu cache: {str(e)}", exc_info=True)
        return result

    def _save_pickle_file(self, data: Dict[str, Any]):
        with open(self.embeddings_file, "wb") as f:
            pickle.dump(data, f)

    async def _semantic_score(self, text: str) -> float:
        if not text.strip():
            return 0.0
        try:
            text_emb = await asyncio.to_thread(self.model.embed_query, text)
            text_emb = np.array(text_emb).reshape(1, -1)
            similarities = cosine_similarity(text_emb, self.embeddings["embeddings"])[0]
            max_sim = float(np.max(similarities))
            top_sim = float(np.mean(np.sort(similarities)[-5:]))
            cat_scores = []
            weights = self.keywords.get(
                "weights",
                {"primary": 2.0, "medical": 1.5, "treatments": 1.8, "symptoms": 1.2},
            )
            for cat_name, cat_emb in self.embeddings["category_embeddings"].items():
                cat_sim = cosine_similarity(text_emb, cat_emb.reshape(1, -1))[0][0]
                category = cat_name.split("_")[-1]
                weight = weights.get(category, 1.0)
                cat_scores.append(cat_sim * weight)
            if cat_scores:
                combined = max_sim * 0.4 + top_sim * 0.3 + max(cat_scores) * 0.3
            else:
                combined = max_sim * 0.6 + top_sim * 0.4
            return max(0.0, min(1.0, combined))
        except Exception as e:
            logger.error(f"Lỗi tính semantic score: {str(e)}", exc_info=True)
            return 0.0

    def _keyword_score(self, text: str) -> float:
        if not text.strip():
            return 0.0
        text_lower = text.lower()
        word_count = len(text.split())
        if word_count == 0:
            return 0.0
        total_score = 0.0
        weights = self.keywords.get(
            "weights",
            {"primary": 2.0, "medical": 1.5, "treatments": 1.8, "symptoms": 1.2},
        )
        matched = set()
        for lang in ["vietnamese", "english"]:
            for category, words in self.keywords[lang].items():
                weight = weights.get(category, 1.0)
                for word in words:
                    if word.lower() not in matched:
                        if len(word.split()) == 1:
                            pattern = r"\b" + re.escape(word.lower()) + r"\b"
                        else:
                            pattern = re.escape(word.lower())
                        matches = len(re.findall(pattern, text_lower))
                        if matches > 0:
                            matched.add(word.lower())
                            score = matches * weight * (1.0 / (1.0 + 0.05 * matches))
                            total_score += score
        if word_count <= 10:
            normalized = total_score / max(word_count * 0.5, 1)
        else:
            normalized = total_score / max(word_count * 0.8, 1)
        return max(0.0, min(1.0, normalized * 1.5))

    async def calculate_diabetes_score(self, text: str) -> float:
        if not text.strip():
            return 0.0
        await self._ensure_initialized()
        semantic_task = self._semantic_score(text)
        keyword_task = asyncio.to_thread(self._keyword_score, text)
        semantic, keyword = await asyncio.gather(semantic_task, keyword_task)

        word_count = len(text.split())

        weight_sem = 0.7
        weight_key = 0.3

        # Bonus cho câu cả semantic và keyword tốt
        if semantic > 0.7 and keyword > 0.3:
            bonus = 1.12
        elif semantic > 0.5 and keyword > 0.2:
            bonus = 1.05
        else:
            bonus = 1.0

        # Penalty cực mạnh cho câu không liên quan
        if keyword < 0.1 and semantic < 0.7:
            penalty = 0.13
        elif keyword < 0.15 and semantic < 0.6:
            penalty = 0.25
        elif keyword < 0.2 and semantic < 0.4:
            penalty = 0.18
        else:
            penalty = 1.0

        final_score = (semantic * weight_sem + keyword * weight_key) * bonus * penalty

        # Phạt câu quá ngắn
        if word_count < 7:
            final_score *= 0.85

        # Phạt câu dài mà ít keyword
        if word_count > 15 and keyword < 0.3:
            final_score *= 0.8

        return round(max(0.0, min(1.0, final_score)), 3)

    async def get_detailed_analysis(self, text: str) -> DiabetesAnalysisResult:
        if not text.strip():
            return DiabetesAnalysisResult(0.0, 0.0, 0.0, "Low", 0)
        await self._ensure_initialized()
        semantic_task = self._semantic_score(text)
        keyword_task = asyncio.to_thread(self._keyword_score, text)
        semantic, keyword = await asyncio.gather(semantic_task, keyword_task)
        final = await self.calculate_diabetes_score(text)
        if final >= self.high_threshold:
            level = "High"
        elif final >= self.medium_threshold:
            level = "Medium"
        else:
            level = "Low"
        return DiabetesAnalysisResult(
            final_score=final,
            semantic_score=round(semantic, 3),
            keyword_score=round(keyword, 3),
            relevance_level=level,
            word_count=len(text.split()),
        )

    async def get_detailed_analysis_dict(self, text: str) -> Dict[str, Any]:
        result = await self.get_detailed_analysis(text)
        return result.to_dict()

    async def score_document_chunks(self, documents: List[Document]) -> List[float]:
        await self._ensure_initialized()
        tasks = [self.calculate_diabetes_score(doc.page_content) for doc in documents]
        return await asyncio.gather(*tasks)

    async def get_overall_document_score(self, documents: List[Document]) -> float:
        scores = await self.score_document_chunks(documents)
        if not scores:
            return 0.0
        relevant_scores = [s for s in scores if s > 0.2]
        if not relevant_scores:
            return max(scores)
        if any(s >= self.high_threshold for s in relevant_scores):
            weights = [s**1.2 for s in relevant_scores]
        else:
            weights = [max(0.2, s) for s in relevant_scores]
        weighted_avg = sum(s * w for s, w in zip(relevant_scores, weights)) / sum(
            weights
        )
        if len(relevant_scores) >= 3:
            weighted_avg *= 1.05
        return round(min(1.0, weighted_avg), 3)


_scorer_instance: Optional[DiabetesScorer] = None
_scorer_lock = threading.Lock()


async def get_scorer_async(
    data_dir: str = "shared", model_dir: str = "model"
) -> DiabetesScorer:
    global _scorer_instance
    if _scorer_instance is not None and _scorer_instance._initialized:
        return _scorer_instance
    with _scorer_lock:
        if _scorer_instance is None:
            _scorer_instance = DiabetesScorer(data_dir, model_dir)
    await _scorer_instance._ensure_initialized()
    return _scorer_instance


async def get_scorer(
    data_dir: str = "shared", model_dir: str = "model"
) -> DiabetesScorer:
    global _scorer_instance
    if _scorer_instance is not None and _scorer_instance._initialized:
        return _scorer_instance
    with _scorer_lock:
        if _scorer_instance is None:
            _scorer_instance = DiabetesScorer(data_dir, model_dir)
        await _scorer_instance._ensure_initialized()
    return _scorer_instance


async def async_score_diabetes_content_with_embedding(
    text: str, data_dir: str = "shared", model_dir: str = "model"
) -> float:
    scorer = await get_scorer_async(data_dir, model_dir)
    return await scorer.calculate_diabetes_score(text)


async def async_analyze_diabetes_content(
    text: str, data_dir: str = "shared", model_dir: str = "model"
) -> DiabetesAnalysisResult:
    scorer = await get_scorer_async(data_dir, model_dir)
    return await scorer.get_detailed_analysis(text)


async def async_analyze_diabetes_content_dict(
    text: str, data_dir: str = "shared", model_dir: str = "model"
) -> Dict[str, Any]:
    scorer = await get_scorer_async(data_dir, model_dir)
    return await scorer.get_detailed_analysis_dict(text)


def analyze_diabetes_content(
    text: str, data_dir: str = "shared", model_dir: str = "model"
) -> DiabetesAnalysisResult:
    scorer = asyncio.run(get_scorer(data_dir, model_dir))
    return asyncio.run(scorer.get_detailed_analysis(text))


def analyze_diabetes_content_dict(
    text: str, data_dir: str = "shared", model_dir: str = "model"
) -> Dict[str, Any]:
    result = analyze_diabetes_content(text, data_dir, model_dir)
    return result.to_dict()


DiabetesScorerUtils = DiabetesScorer
async_get_scorer = get_scorer_async
if __name__ == "__main__":
    print("DIABETES CONTENT SCORING - TEST")
    print("=" * 50)

    test_texts = [
        "Bệnh tiểu đường type 2 là bệnh mãn tính. Bệnh nhân có triệu chứng khát nước và đi tiểu nhiều.",
        "Diabetes mellitus affects glucose metabolism. Treatment includes insulin therapy.",
        "Hôm nay trời đẹp, tôi đi chơi công viên với bạn bè. và gặp bạn bè tiểu đường",
        "The patient presented with elevated HbA1c levels.",
        "Biến chứng tiểu đường có thể ảnh hưởng nghiêm trọng đến sức khỏe.",
    ]

    async def test_async():
        print("Testing ASYNC version...")
        scorer = await get_scorer_async()

        for i, text in enumerate(test_texts, 1):
            print(f"\nTest {i}: {text[:60]}...")
            analysis = await scorer.get_detailed_analysis(text)
            # print(f"Score: {analysis.final_score} ({analysis.relevance_level})")
            print(
                f"Semantic: {analysis.final_score}, Keyword: {analysis.keyword_score}"
            )
            # print(f"Object: {analysis}")

    try:
        # Test async version
        asyncio.run(test_async())

    except Exception as e:
        print(f"Error: {e}")
