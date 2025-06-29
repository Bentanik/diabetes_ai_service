# -*- coding: utf-8 -*-
"""Test simple cho Qdrant RAG API."""

import requests
import json


def test_upload_and_search():
    """Test upload text và search."""
    print("🧪 Testing Qdrant RAG API...")

    # 1. Upload test text
    print("\n1️⃣ Uploading test text...")
    text_data = {
        "text": "Qdrant là vector database hiệu suất cao cho machine learning và AI applications. Nó hỗ trợ similarity search và vector operations rất nhanh.",
        "metadata": {"type": "test_doc", "source": "manual"},
    }

    upload_response = requests.post(
        "http://localhost:8000/api/v1/chatbot/rag/upload_text", json=text_data
    )

    print(f"Upload Status: {upload_response.status_code}")
    if upload_response.status_code == 200:
        upload_data = upload_response.json()
        print(f"✅ Upload success: {upload_data['chunks_added']} chunks added")
    else:
        print(f"❌ Upload failed: {upload_response.text}")
        return

    # 2. Test search
    print("\n2️⃣ Testing search...")
    search_data = {"question": "vector database", "k": 3, "method": "hybrid"}

    search_response = requests.post(
        "http://localhost:8000/api/v1/chatbot/rag/search", json=search_data
    )

    print(f"Search Status: {search_response.status_code}")
    if search_response.status_code == 200:
        search_data = search_response.json()
        print(f"✅ Search success:")
        print(f"   Documents found: {search_data['num_sources']}")
        print(f"   Search method: {search_data['search_method']}")
        print(f"   Confidence: {search_data['confidence_score']:.3f}")

        if search_data["sources"]:
            for i, source in enumerate(search_data["sources"][:2]):
                print(
                    f"   📄 Result {i+1}: score={source.get('similarity_score', 0):.3f}"
                )
    else:
        print(f"❌ Search failed: {search_response.text}")


if __name__ == "__main__":
    test_upload_and_search()
