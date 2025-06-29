# -*- coding: utf-8 -*-
"""Final test for complete RAG pipeline."""

import requests


def test_complete_pipeline():
    """Test complete RAG: retrieval + LLM generation."""
    print("🚀 Testing Complete RAG Pipeline...")

    # Test full query với LLM generation
    query_data = {
        "question": "Qdrant có ưu điểm gì?",
        "k": 3,
        "method": "hybrid",
        "vietnamese_prompt": True,
    }

    response = requests.post(
        "http://localhost:8000/api/v1/chatbot/rag/query", json=query_data
    )

    print("🤖 Full Query Test:")
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f'✅ Success: {result["success"]}')
        print(f'📊 Sources: {result["num_sources"]}')
        print(f'🎯 Confidence: {result["confidence_score"]:.3f}')
        print(f'🔍 Method: {result["retrieval_method"]}')

        if result["success"] and result["answer"]:
            answer = result["answer"]
            if len(answer) > 200:
                answer = answer[:200] + "..."
            print(f"💬 Answer: {answer}")

            # Sources info
            if result["sources"]:
                print(f"📚 Sources Details:")
                for i, source in enumerate(result["sources"][:2]):
                    score = source.get("similarity_score", 0)
                    content = source.get("content_preview", "")[:60]
                    print(f"   {i+1}. Score: {score:.3f} | {content}...")
        else:
            print("❌ No answer generated")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)


if __name__ == "__main__":
    test_complete_pipeline()
