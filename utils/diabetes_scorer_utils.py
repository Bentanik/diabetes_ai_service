"""
Diabetes Content Scoring

Chức năng:
- Tính điểm liên quan đến diabetes cho văn bản
- Hỗ trợ tiếng Việt và tiếng Anh
- Kết hợp semantic similarity và keyword matching
- Async support và thread-safe initialization
"""

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
from core.llm import get_embedding_model


class DiabetesAnalysisResult:
    """
    Kết quả phân tích diabetes với property access
    """

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
    """
    Class chính để scoring nội dung diabetes
    """

    def __init__(self, data_dir: str = "shared", model_dir: str = "model"):
        """Khởi tạo scorer - chỉ setup paths, không load model ngay"""
        self.data_dir = Path.cwd() / data_dir
        self.model_dir = Path.cwd() / model_dir
        self.keywords_file = self.data_dir / "diabetes_words.json"
        self.embeddings_file = self.model_dir / "diabetes_embeddings.pkl"

        # Tạo thư mục nếu chưa có
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        # State variables - lazy loading
        self.model = None
        self.keywords = None
        self.embeddings = None
        self._initialized = False
        self._initializing = False
        self._init_lock = asyncio.Lock()

        # Ngưỡng phân loại
        self.high_threshold = 0.6
        self.medium_threshold = 0.35

    async def _ensure_initialized(self):
        """Đảm bảo model đã được khởi tạo - thread-safe"""
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
                print("Khởi tạo DiabetesScorer...")

                # Load model trong thread pool
                self.model = await asyncio.to_thread(get_embedding_model)
                self.keywords = await asyncio.to_thread(self._load_keywords)
                self.embeddings = await self._load_embeddings()

                self._initialized = True
                print("DiabetesScorer đã sẵn sàng!")

            except Exception as e:
                print(f"Lỗi khởi tạo DiabetesScorer: {e}")
                raise
            finally:
                self._initializing = False

    def _load_keywords(self) -> Dict[str, Any]:
        """Load hoặc tạo keywords mặc định"""
        if self.keywords_file.exists():
            try:
                with open(self.keywords_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                pass

        # Keywords mặc định
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

        # Lưu file
        try:
            with open(self.keywords_file, "w", encoding="utf-8") as f:
                json.dump(keywords, f, ensure_ascii=False, indent=2)
        except:
            pass

        return keywords

    async def _load_embeddings(self) -> Dict[str, Any]:
        """Load embeddings từ cache hoặc tạo mới"""
        if self.embeddings_file.exists():
            try:
                data = await asyncio.to_thread(self._load_pickle_file)
                if data and "embeddings" in data and len(data["embeddings"]) > 0:
                    print(f"Loaded {len(data['texts'])} embeddings từ cache")
                    return data
            except Exception as e:
                print(f"Lỗi load cache: {e}")

        return await self._create_embeddings()

    def _load_pickle_file(self) -> Dict[str, Any]:
        """Helper để load pickle file"""
        with open(self.embeddings_file, "rb") as f:
            return pickle.load(f)

    async def _create_embeddings(self) -> Dict[str, Any]:
        """Tạo embeddings mới"""
        print("Đang tạo embeddings...")
        start_time = time.time()

        # Thu thập tất cả keywords
        all_texts = []
        for lang in ["vietnamese", "english"]:
            for category, words in self.keywords[lang].items():
                all_texts.extend([w for w in words if isinstance(w, str)])

        # Loại bỏ duplicate
        unique_texts = list(set(all_texts))
        print(f"Tạo embeddings cho {len(unique_texts)} từ khóa...")

        # Tạo embeddings trong thread pool
        embeddings = await asyncio.to_thread(self.model.encode, unique_texts)

        # Tạo category embeddings
        category_embeddings = {}
        for lang in ["vietnamese", "english"]:
            for category, words in self.keywords[lang].items():
                if words:
                    cat_embeddings = await asyncio.to_thread(self.model.encode, words)
                    category_embeddings[f"{lang}_{category}"] = np.mean(
                        cat_embeddings, axis=0
                    )

        result = {
            "texts": unique_texts,
            "embeddings": embeddings,
            "mean_embedding": np.mean(embeddings, axis=0),
            "category_embeddings": category_embeddings,
        }

        # Lưu cache
        try:
            await asyncio.to_thread(self._save_pickle_file, result)
            print(f"Embeddings đã tạo và cache trong {time.time() - start_time:.1f}s")
        except Exception as e:
            print(f"Không thể lưu cache: {e}")

        return result

    def _save_pickle_file(self, data: Dict[str, Any]):
        """Helper để save pickle file"""
        with open(self.embeddings_file, "wb") as f:
            pickle.dump(data, f)

    async def _semantic_score(self, text: str) -> float:
        """Tính điểm semantic similarity"""
        if not text.strip():
            return 0.0

        try:
            # Encode text trong thread pool
            text_emb = await asyncio.to_thread(self.model.encode, [text])
            similarities = cosine_similarity(text_emb, self.embeddings["embeddings"])[0]

            # Kết hợp các metrics
            max_sim = float(np.max(similarities))
            top_sim = float(np.mean(np.sort(similarities)[-5:]))  # Top 5

            # Category similarities với weights
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

            # Kết hợp scores
            if cat_scores:
                combined = max_sim * 0.4 + top_sim * 0.3 + max(cat_scores) * 0.3
            else:
                combined = max_sim * 0.6 + top_sim * 0.4

            return max(0.0, min(1.0, combined))
        except:
            return 0.0

    def _keyword_score(self, text: str) -> float:
        """Tính điểm keyword matching"""
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

        # Duyệt keywords
        for lang in ["vietnamese", "english"]:
            for category, words in self.keywords[lang].items():
                weight = weights.get(category, 1.0)
                for word in words:
                    if word.lower() not in matched:
                        # Regex pattern
                        if len(word.split()) == 1:
                            pattern = r"\b" + re.escape(word.lower()) + r"\b"
                        else:
                            pattern = re.escape(word.lower())

                        matches = len(re.findall(pattern, text_lower))
                        if matches > 0:
                            matched.add(word.lower())
                            # Diminishing returns
                            score = matches * weight * (1.0 / (1.0 + 0.05 * matches))
                            total_score += score

        # Normalize theo độ dài text
        if word_count <= 10:
            normalized = total_score / max(word_count * 0.5, 1)
        else:
            normalized = total_score / max(word_count * 0.8, 1)

        return max(0.0, min(1.0, normalized * 1.5))

    async def calculate_diabetes_score(self, text: str) -> float:
        """Tính điểm diabetes tổng hợp"""
        if not text.strip():
            return 0.0

        # Đảm bảo đã khởi tạo
        await self._ensure_initialized()

        # Chạy song song semantic và keyword scoring
        semantic_task = self._semantic_score(text)
        keyword_task = asyncio.to_thread(self._keyword_score, text)

        semantic, keyword = await asyncio.gather(semantic_task, keyword_task)

        # Logic kết hợp
        if keyword > 0.7 and semantic > 0.3:
            combined = semantic * 0.4 + keyword * 0.6
        elif keyword > 0.3 and semantic > 0.2:
            combined = semantic * 0.6 + keyword * 0.4
        else:
            if keyword == 0.0 and semantic > 0.5:
                combined = semantic * 0.3
            else:
                combined = semantic * 0.7 + keyword * 0.3

        # Điều chỉnh theo độ dài
        word_count = len(text.split())
        if word_count < 5:
            combined *= 0.9
        elif word_count > 50:
            combined *= 1.02

        return round(max(0.0, min(1.0, combined)), 3)

    def calculate_diabetes_score_sync(self, text: str) -> float:
        if not text.strip():
            return 0.0

        if not self._initialized:
            raise RuntimeError(
                "DiabetesScorer chưa được khởi tạo. Hãy dùng async version."
            )

        semantic = asyncio.run(self._semantic_score(text))
        keyword = self._keyword_score(text)

        # Logic kết hợp giống async version
        if keyword > 0.7 and semantic > 0.3:
            combined = semantic * 0.4 + keyword * 0.6
        elif keyword > 0.3 and semantic > 0.2:
            combined = semantic * 0.6 + keyword * 0.4
        else:
            if keyword == 0.0 and semantic > 0.5:
                combined = semantic * 0.3
            else:
                combined = semantic * 0.7 + keyword * 0.3

        word_count = len(text.split())
        if word_count < 5:
            combined *= 0.9
        elif word_count > 50:
            combined *= 1.02

        return round(max(0.0, min(1.0, combined)), 3)

    async def get_detailed_analysis(self, text: str) -> DiabetesAnalysisResult:
        """Phân tích chi tiết - trả về object có thể truy cập properties"""
        if not text.strip():
            return DiabetesAnalysisResult(0.0, 0.0, 0.0, "Low", 0)

        await self._ensure_initialized()

        # Chạy song song
        semantic_task = self._semantic_score(text)
        keyword_task = asyncio.to_thread(self._keyword_score, text)

        semantic, keyword = await asyncio.gather(semantic_task, keyword_task)
        final = await self.calculate_diabetes_score(text)

        # Xác định level
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
        """Phân tích chi tiết - trả về dict để backward compatibility"""
        result = await self.get_detailed_analysis(text)
        return result.to_dict()

    async def score_document_chunks(self, documents: List[Document]) -> List[float]:
        """Score nhiều documents song song"""
        await self._ensure_initialized()
        tasks = [self.calculate_diabetes_score(doc.page_content) for doc in documents]
        return await asyncio.gather(*tasks)

    async def get_overall_document_score(self, documents: List[Document]) -> float:
        """Điểm tổng hợp cho document"""
        scores = await self.score_document_chunks(documents)
        if not scores:
            return 0.0

        relevant_scores = [s for s in scores if s > 0.2]
        if not relevant_scores:
            return max(scores)

        # Weighted average
        if any(s >= self.high_threshold for s in relevant_scores):
            weights = [s**1.2 for s in relevant_scores]
        else:
            weights = [max(0.2, s) for s in relevant_scores]

        weighted_avg = sum(s * w for s, w in zip(relevant_scores, weights)) / sum(
            weights
        )

        # Bonus cho nhiều chunks relevant
        if len(relevant_scores) >= 3:
            weighted_avg *= 1.05

        return round(min(1.0, weighted_avg), 3)


# Thread-safe singleton
_scorer_instance: Optional[DiabetesScorer] = None
_scorer_lock = threading.Lock()


async def get_scorer_async(
    data_dir: str = "shared", model_dir: str = "model"
) -> DiabetesScorer:
    """Lấy singleton scorer - async và thread-safe"""
    global _scorer_instance

    if _scorer_instance is not None and _scorer_instance._initialized:
        return _scorer_instance

    # Thread-safe singleton creation
    with _scorer_lock:
        if _scorer_instance is None:
            _scorer_instance = DiabetesScorer(data_dir, model_dir)

    # Async initialization
    await _scorer_instance._ensure_initialized()
    return _scorer_instance


def get_scorer(data_dir: str = "shared", model_dir: str = "model") -> DiabetesScorer:
    global _scorer_instance

    with _scorer_lock:
        if _scorer_instance is None:
            _scorer_instance = DiabetesScorer(data_dir, model_dir)
            print("CẢNH BÁO: Đang khởi tạo sync - có thể block requests!")
            _scorer_instance.model = get_embedding_model()
            _scorer_instance.keywords = _scorer_instance._load_keywords()

            # Kiểm tra nếu đang trong event loop
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    raise RuntimeError(
                        "Không thể khởi tạo sync DiabetesScorer trong async context! "
                        "Hãy dùng get_scorer_async() thay thế."
                    )
            except RuntimeError as e:
                if "no running event loop" in str(e):
                    # Không có event loop đang chạy, OK để dùng asyncio.run()
                    pass
                else:
                    # Có event loop đang chạy, không được dùng asyncio.run()
                    raise e

            # Chỉ chạy nếu không có event loop
            _scorer_instance.embeddings = asyncio.run(
                _scorer_instance._load_embeddings()
            )
            _scorer_instance._initialized = True
            print("Sync initialization hoàn tất")

    return _scorer_instance


# Convenience functions - ưu tiên async
async def async_score_diabetes_content_with_embedding(
    text: str, data_dir: str = "shared", model_dir: str = "model"
) -> float:
    """Hàm tiện ích tính điểm diabetes - async (khuyến khích)"""
    scorer = await get_scorer_async(data_dir, model_dir)
    return await scorer.calculate_diabetes_score(text)


async def async_analyze_diabetes_content(
    text: str, data_dir: str = "shared", model_dir: str = "model"
) -> DiabetesAnalysisResult:
    """Phân tích chi tiết diabetes content - async (khuyến khích) - trả về object"""
    scorer = await get_scorer_async(data_dir, model_dir)
    return await scorer.get_detailed_analysis(text)


async def async_analyze_diabetes_content_dict(
    text: str, data_dir: str = "shared", model_dir: str = "model"
) -> Dict[str, Any]:
    """Phân tích chi tiết diabetes content - async - trả về dict để backward compatibility"""
    scorer = await get_scorer_async(data_dir, model_dir)
    return await scorer.get_detailed_analysis_dict(text)


def score_diabetes_content_with_embedding(
    text: str, data_dir: str = "shared", model_dir: str = "model"
) -> float:
    """Hàm tiện ích tính điểm diabetes - sync có thể block"""
    scorer = get_scorer(data_dir, model_dir)
    if scorer._initialized:
        return scorer.calculate_diabetes_score_sync(text)
    else:
        return asyncio.run(scorer.calculate_diabetes_score(text))


def analyze_diabetes_content(
    text: str, data_dir: str = "shared", model_dir: str = "model"
) -> DiabetesAnalysisResult:
    """Phân tích chi tiết diabetes content - sync có thể block - trả về object"""
    scorer = get_scorer(data_dir, model_dir)
    if scorer._initialized:
        semantic = asyncio.run(scorer._semantic_score(text))
        keyword = scorer._keyword_score(text)
        final = scorer.calculate_diabetes_score_sync(text)

        if final >= scorer.high_threshold:
            level = "High"
        elif final >= scorer.medium_threshold:
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
    else:
        return asyncio.run(scorer.get_detailed_analysis(text))


def analyze_diabetes_content_dict(
    text: str, data_dir: str = "shared", model_dir: str = "model"
) -> Dict[str, Any]:
    """Phân tích chi tiết diabetes content - sync - trả về dict để backward compatibility"""
    result = analyze_diabetes_content(text, data_dir, model_dir)
    return result.to_dict()


# Backward compatibility
DiabetesScorerUtils = DiabetesScorer
async_get_scorer = get_scorer_async

if __name__ == "__main__":
    print("DIABETES CONTENT SCORING - TEST")
    print("=" * 50)

    test_texts = [
        "Bệnh tiểu đường type 2 là bệnh mãn tính. Bệnh nhân có triệu chứng khát nước và đi tiểu nhiều.",
        "Diabetes mellitus affects glucose metabolism. Treatment includes insulin therapy.",
        "Hôm nay trời đẹp, tôi đi chơi công viên với bạn bè.",
        "The patient presented with elevated HbA1c levels.",
        "Biến chứng tiểu đường có thể ảnh hưởng nghiêm trọng đến sức khỏe.",
    ]

    async def test_async():
        print("Testing ASYNC version...")
        scorer = await get_scorer_async()

        for i, text in enumerate(test_texts, 1):
            print(f"\nTest {i}: {text[:60]}...")
            # Test new object-based result
            analysis = await scorer.get_detailed_analysis(text)
            print(f"Score: {analysis.final_score} ({analysis.relevance_level})")
            print(
                f"Semantic: {analysis.semantic_score}, Keyword: {analysis.keyword_score}"
            )
            print(f"Object: {analysis}")

    def test_sync():
        print("Testing SYNC version...")
        scorer = get_scorer()

        for i, text in enumerate(test_texts[:2], 1):
            print(f"\nSync Test {i}: {text[:60]}...")
            # Test new object-based result
            analysis = analyze_diabetes_content(text)
            print(f"Score: {analysis.final_score} ({analysis.relevance_level})")
            print(
                f"Direct access: final_score={analysis.final_score}, level={analysis.relevance_level}"
            )

    try:
        # Test async version
        asyncio.run(test_async())

        print("\n" + "=" * 50)
        test_sync()

    except Exception as e:
        print(f"Error: {e}")
