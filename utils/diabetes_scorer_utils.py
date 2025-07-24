import json
import os
from pathlib import Path
import re
from typing import List, Dict, Any, Optional
from langchain.schema import Document
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import time
import asyncio
from core.llm import get_embedding_model  # Updated import to match your code


class DiabetesScorerUtils:
    def __init__(self, data_dir: str = "shared", model_dir: str = "model"):
        """Initialize with diabetes-specific embeddings using fixed directories."""
        self.data_dir = Path.cwd() / data_dir
        self.model_dir = Path.cwd() / model_dir
        self.keywords_file = self.data_dir / "diabetes_words.json"
        self.embeddings_cache_file = self.model_dir / "diabetes_embeddings.pkl"

        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        print(f"Keywords file: {self.keywords_file}")
        print(f"Embeddings cache: {self.embeddings_cache_file}")

        self.model = get_embedding_model()
        self.diabetes_keywords = self._load_keywords()
        self.diabetes_embeddings = self._load_or_create_embeddings()

        self.high_threshold = 0.6
        self.medium_threshold = 0.35

    def _load_keywords(self) -> Dict[str, Any]:
        """Load keywords from JSON file"""
        try:
            if not self.keywords_file.exists():
                print(f"Creating default keywords file at {self.keywords_file}")
                self._create_default_keywords()

            with open(self.keywords_file, "r", encoding="utf-8") as f:
                keywords = json.load(f)
                print(f"Loaded {len(keywords)} keyword categories")
                return keywords
        except Exception as e:
            print(f"Error loading keywords from {self.keywords_file}: {e}")
            print("Creating default keywords...")
            return self._create_default_keywords()

    def _create_default_keywords(self) -> Dict[str, Any]:
        """Create default keywords if file doesn't exist"""
        default_keywords = {
            "vietnamese": {
                "primary_keywords": [
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
                "medical_terms": [
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
                "primary_keywords": [
                    "diabetes",
                    "diabetes mellitus",
                    "type 1 diabetes",
                    "type 2 diabetes",
                    "blood glucose",
                    "insulin",
                    "HbA1c",
                ],
                "medical_terms": [
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
            "keyword_weights": {
                "primary_keywords": 2.0,
                "medical_terms": 1.5,
                "treatments": 1.8,
                "symptoms": 1.2,
            },
        }

        try:
            with open(self.keywords_file, "w", encoding="utf-8") as f:
                json.dump(default_keywords, f, ensure_ascii=False, indent=2)
            print(f"Created default keywords file at {self.keywords_file}")
        except Exception as e:
            print(f"Error creating keywords file: {e}")

        return default_keywords

    async def _async_encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts in a non-blocking manner using a thread."""
        return await asyncio.to_thread(self.model.encode, texts, show_progress_bar=True)

    def _load_or_create_embeddings(self) -> Dict[str, Any]:
        """Load cached embeddings or create new ones with better validation"""
        if self.embeddings_cache_file.exists():
            try:
                print(f"Loading cached embeddings from {self.embeddings_cache_file}")
                with open(self.embeddings_cache_file, "rb") as f:
                    embeddings_data = pickle.load(f)

                required_keys = [
                    "texts",
                    "embeddings",
                    "mean_embedding",
                    "category_embeddings",
                ]
                if (
                    isinstance(embeddings_data, dict)
                    and all(key in embeddings_data for key in required_keys)
                    and len(embeddings_data["texts"]) > 0
                ):
                    print(
                        f"Successfully loaded {len(embeddings_data['texts'])} cached embeddings"
                    )
                    return embeddings_data
                else:
                    print("Invalid embeddings structure, recreating...")

            except Exception as e:
                print(f"Error loading cached embeddings: {e}")
                print("Recreating embeddings...")

        return self._create_diabetes_embeddings()

    async def precompute_embeddings(self) -> None:
        """Precompute and cache embeddings at startup."""
        self.diabetes_embeddings = await self._async_create_diabetes_embeddings()

    async def _async_create_diabetes_embeddings(self) -> Dict[str, Any]:
        """Create embeddings for diabetes-related terms and phrases asynchronously"""
        print("Creating diabetes embeddings...")
        start_time = time.time()

        diabetes_texts = []
        for lang in ["vietnamese", "english"]:
            lang_keywords = self.diabetes_keywords.get(lang, {})
            for category, keywords in lang_keywords.items():
                if isinstance(keywords, list):
                    diabetes_texts.extend([k for k in keywords if isinstance(k, str)])

        context_patterns = self.diabetes_keywords.get("context_patterns", {})
        for lang, patterns in context_patterns.items():
            if isinstance(patterns, list):
                diabetes_texts.extend([p for p in patterns if isinstance(p, str)])

        seen = set()
        unique_texts = []
        for text in diabetes_texts:
            if text.lower() not in seen:
                seen.add(text.lower())
                unique_texts.append(text)

        if not unique_texts:
            raise ValueError("No diabetes texts found to create embeddings")

        print(f"Creating embeddings for {len(unique_texts)} unique texts...")

        batch_size = 32
        all_embeddings = []
        for i in range(0, len(unique_texts), batch_size):
            batch = unique_texts[i : i + batch_size]
            batch_embeddings = await self._async_encode(batch)
            all_embeddings.append(batch_embeddings)
            print(
                f"Processed batch {i//batch_size + 1}/{(len(unique_texts)-1)//batch_size + 1}"
            )

        embeddings = np.vstack(all_embeddings)

        print("Creating category embeddings...")
        category_embeddings = {}
        for lang in ["vietnamese", "english"]:
            lang_keywords = self.diabetes_keywords.get(lang, {})
            for category, keywords in lang_keywords.items():
                if isinstance(keywords, list) and keywords:
                    try:
                        valid_keywords = [
                            k for k in keywords if isinstance(k, str) and k.strip()
                        ]
                        if valid_keywords:
                            cat_embeddings = await self._async_encode(valid_keywords)
                            category_embeddings[f"{lang}_{category}"] = np.mean(
                                cat_embeddings, axis=0
                            )
                    except Exception as e:
                        print(
                            f"Warning: Error creating embeddings for {lang}_{category}: {e}"
                        )

        embedding_dict = {
            "texts": unique_texts,
            "embeddings": embeddings,
            "mean_embedding": np.mean(embeddings, axis=0),
            "category_embeddings": category_embeddings,
            "creation_time": time.time(),
            "model_name": str(self.model.get_sentence_embedding_dimension()),
        }

        try:
            with open(self.embeddings_cache_file, "wb") as f:
                pickle.dump(embedding_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            elapsed = time.time() - start_time
            print(f"Embeddings created and cached in {elapsed:.2f} seconds")
            print(f"Cache saved to: {self.embeddings_cache_file}")
        except Exception as e:
            print(f"Warning: Could not cache embeddings: {e}")
            print("Embeddings will be recreated on next run")

        return embedding_dict

    def _calculate_semantic_similarity(self, text: str) -> float:
        """Enhanced semantic similarity calculation with proper bounds"""
        if not text or not text.strip():
            return 0.0

        try:
            text_embedding = self.model.encode([text])
            diabetes_embeddings = self.diabetes_embeddings["embeddings"]
            similarities = cosine_similarity(text_embedding, diabetes_embeddings)[0]
            similarities = np.clip(similarities, -1.0, 1.0)

            max_similarity = float(np.max(similarities))
            top_similarity = float(
                np.mean(np.sort(similarities)[-min(10, len(similarities)) :])
            )
            percentile_90 = float(np.percentile(similarities, 90))

            category_similarities = []
            weights = self.diabetes_keywords.get("keyword_weights", {})
            category_embeddings = self.diabetes_embeddings.get(
                "category_embeddings", {}
            )
            if isinstance(category_embeddings, dict):
                for cat_name, cat_embedding in category_embeddings.items():
                    if isinstance(cat_embedding, np.ndarray):
                        cat_sim = cosine_similarity(
                            text_embedding, cat_embedding.reshape(1, -1)
                        )[0][0]
                        cat_sim = float(np.clip(cat_sim, -1.0, 1.0))
                        category = (
                            cat_name.split("_", 1)[-1] if "_" in cat_name else cat_name
                        )
                        weight = weights.get(category, 1.0)
                        category_similarities.append(cat_sim * weight)

            mean_embedding = self.diabetes_embeddings["mean_embedding"].reshape(1, -1)
            mean_sim = float(
                np.clip(
                    cosine_similarity(text_embedding, mean_embedding)[0][0], -1.0, 1.0
                )
            )

            if category_similarities:
                max_cat_sim = max(category_similarities)
                avg_cat_sim = np.mean(category_similarities)
                combined = (
                    max_similarity * 0.25
                    + top_similarity * 0.25
                    + percentile_90 * 0.15
                    + max_cat_sim * 0.2
                    + avg_cat_sim * 0.1
                    + mean_sim * 0.05
                )
            else:
                combined = (
                    max_similarity * 0.4
                    + top_similarity * 0.35
                    + percentile_90 * 0.15
                    + mean_sim * 0.1
                )

            return float(np.clip(combined, 0.0, 1.0))

        except Exception as e:
            print(f"Error in semantic similarity: {e}")
            return 0.0

    def _calculate_keyword_score(self, text: str) -> float:
        """Enhanced keyword-based scoring"""
        if not text or not text.strip():
            return 0.0

        text_lower = text.lower()
        word_count = len(text.split())
        if word_count == 0:
            return 0.0

        weights = self.diabetes_keywords.get("keyword_weights", {})
        total_score = 0.0
        matched_keywords = set()

        for lang in ["vietnamese", "english"]:
            lang_keywords = self.diabetes_keywords.get(lang, {})
            for category, keywords in lang_keywords.items():
                if isinstance(keywords, list):
                    weight = weights.get(category, 1.0)
                    for keyword in keywords:
                        if (
                            isinstance(keyword, str)
                            and keyword.lower() not in matched_keywords
                        ):
                            if len(keyword.split()) == 1:
                                pattern = r"\b" + re.escape(keyword.lower()) + r"\b"
                            else:
                                pattern = re.escape(keyword.lower())
                            matches = len(re.findall(pattern, text_lower))
                            if matches > 0:
                                matched_keywords.add(keyword.lower())
                                keyword_score = (
                                    matches * weight * (1.0 / (1.0 + 0.05 * matches))
                                )
                                total_score += keyword_score

        if word_count <= 10:
            normalized = total_score / max(word_count * 0.5, 1)
        else:
            normalized = total_score / max(word_count * 0.8, 1)

        return max(0.0, min(1.0, normalized * 1.5))

    def calculate_diabetes_score(self, text: str) -> float:
        """Calculate comprehensive diabetes relevance score"""
        if not text or not text.strip():
            return 0.0

        semantic_score = max(0.0, min(1.0, self._calculate_semantic_similarity(text)))
        keyword_score = max(0.0, min(1.0, self._calculate_keyword_score(text)))

        if keyword_score > 0.7 and semantic_score > 0.3:
            combined = semantic_score * 0.4 + keyword_score * 0.6
        elif keyword_score > 0.3 and semantic_score > 0.2:
            combined = semantic_score * 0.6 + keyword_score * 0.4
        else:
            if keyword_score == 0.0 and semantic_score > 0.5:
                combined = semantic_score * 0.3
            else:
                combined = semantic_score * 0.7 + keyword_score * 0.3

        word_count = len(text.split())
        if word_count < 5:
            combined *= 0.9
        elif word_count > 50:
            combined *= 1.02

        return round(max(0.0, min(1.0, combined)), 3)

    async def async_calculate_diabetes_score(self, text: str) -> float:
        """Calculate diabetes score asynchronously"""
        return self.calculate_diabetes_score(text)

    def get_detailed_analysis(self, text: str) -> Dict[str, Any]:
        """Get detailed analysis"""
        semantic_score = self._calculate_semantic_similarity(text)
        keyword_score = self._calculate_keyword_score(text)
        final_score = self.calculate_diabetes_score(text)

        similar_concepts = []
        if text.strip():
            try:
                text_embedding = self.model.encode([text])
                similarities = cosine_similarity(
                    text_embedding, self.diabetes_embeddings["embeddings"]
                )[0]
                similarities = np.clip(similarities, -1.0, 1.0)
                top_indices = np.argsort(similarities)[-5:][::-1]
                similar_concepts = [
                    {
                        "concept": self.diabetes_embeddings["texts"][i],
                        "similarity": float(similarities[i]),
                    }
                    for i in top_indices
                ]
            except Exception as e:
                print(f"Error finding similar concepts: {e}")

        if final_score >= self.high_threshold:
            relevance_level = "High"
        elif final_score >= self.medium_threshold:
            relevance_level = "Medium"
        else:
            relevance_level = "Low"

        return {
            "final_score": final_score,
            "semantic_score": round(max(0.0, min(1.0, semantic_score)), 3),
            "keyword_score": round(max(0.0, min(1.0, keyword_score)), 3),
            "relevance_level": relevance_level,
            "similar_concepts": similar_concepts,
            "word_count": len(text.split()) if text else 0,
        }

    async def async_get_detailed_analysis(self, text: str) -> Dict[str, Any]:
        """Get detailed analysis asynchronously"""
        return self.get_detailed_analysis(text)

    def score_document_chunks(self, documents: List[Document]) -> List[float]:
        """Score multiple document chunks"""
        return [self.calculate_diabetes_score(doc.page_content) for doc in documents]

    async def async_score_document_chunks(
        self, documents: List[Document]
    ) -> List[float]:
        """Score multiple document chunks asynchronously"""
        return await asyncio.gather(
            *[
                self.async_calculate_diabetes_score(doc.page_content)
                for doc in documents
            ]
        )

    def get_overall_document_score(self, documents: List[Document]) -> float:
        """Get overall document score"""
        scores = self.score_document_chunks(documents)
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

    async def async_get_overall_document_score(
        self, documents: List[Document]
    ) -> float:
        """Get overall document score asynchronously"""
        scores = await self.async_score_document_chunks(documents)
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


_scorer_instance: Optional[DiabetesScorerUtils] = None


def get_scorer(
    data_dir: str = "shared", model_dir: str = "model"
) -> DiabetesScorerUtils:
    """Get singleton scorer instance"""
    global _scorer_instance
    if _scorer_instance is None:
        _scorer_instance = DiabetesScorerUtils(data_dir, model_dir)
    return _scorer_instance


async def async_get_scorer(
    data_dir: str = "shared", model_dir: str = "model"
) -> DiabetesScorerUtils:
    """Get singleton scorer instance asynchronously"""
    return get_scorer(data_dir, model_dir)


def score_diabetes_content_with_embedding(
    text: str, data_dir: str = "shared", model_dir: str = "model"
) -> float:
    """Simple function to score diabetes relevance"""
    return get_scorer(data_dir, model_dir).calculate_diabetes_score(text)


async def async_score_diabetes_content_with_embedding(
    text: str, data_dir: str = "shared", model_dir: str = "model"
) -> float:
    """Score diabetes relevance asynchronously"""
    scorer = await async_get_scorer(data_dir, model_dir)
    return await scorer.async_calculate_diabetes_score(text)


def analyze_diabetes_content(
    text: str, data_dir: str = "shared", model_dir: str = "model"
) -> Dict[str, Any]:
    """Get detailed analysis"""
    return get_scorer(data_dir, model_dir).get_detailed_analysis(text)


async def async_analyze_diabetes_content(
    text: str, data_dir: str = "shared", model_dir: str = "model"
) -> Dict[str, Any]:
    """Get detailed analysis asynchronously"""
    scorer = await async_get_scorer(data_dir, model_dir)
    return await scorer.async_get_detailed_analysis(text)


if __name__ == "__main__":
    print("OPTIMIZED DIABETES CONTENT SCORING")
    print("=" * 50)

    test_texts = [
        "Bệnh tiểu đường type 2 là bệnh mãn tính. Bệnh nhân có triệu chứng khát nước và đi tiểu nhiều.",
        "Diabetes mellitus affects glucose metabolism. Treatment includes insulin therapy.",
        "Hôm nay trời đẹp, tôi đi chơi công viên với bạn bè.",
        "The patient presented with elevated HbA1c levels.",
        "Công ty phần mềm phát triển ứng dụng mobile.",
        "Biến chứng tiểu đường có thể ảnh hưởng nghiêm trọng đến sức khỏe.",
    ]

    try:
        scorer = get_scorer()
        for i, text in enumerate(test_texts, 1):
            print(f"\nText {i}: {text[:50]}...")
            analysis = scorer.get_detailed_analysis(text)
            print(
                f"Final Score: {analysis['final_score']} ({analysis['relevance_level']})"
            )
            print(
                f"Semantic: {analysis['semantic_score']}, Keyword: {analysis['keyword_score']}"
            )

    except Exception as e:
        print(f"Error: {e}")
