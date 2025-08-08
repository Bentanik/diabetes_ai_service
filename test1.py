# main.py
import asyncio
from core.embedding.embedding_model import EmbeddingModel
from diabetes_semantic.diabetes_scorer import DiabetesSemanticScorer


async def main():
    # Load model trước
    print("Loading model externally...")
    embedding_model = await EmbeddingModel.get_instance()
    model = embedding_model.model
    scorer = DiabetesSemanticScorer(model=model)
    await scorer.initialize()

    # Test
    result = await scorer.calculate_semantic_score("Bệnh nhân bị tiểu đường tuýp 2")
    print(f"Score: {result.score:.3f}, Level: {result.relevance_level}")

    # Check stats
    stats = scorer.get_statistics()
    print(f"Model provided externally: {stats['model_info']['provided_externally']}")
    print(f"Model name: {stats['model_info']['model_name']}")


if __name__ == "__main__":
    asyncio.run(main())
