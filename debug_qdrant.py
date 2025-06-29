# -*- coding: utf-8 -*-
"""Debug Qdrant raw data."""

import requests
import json
import traceback


def check_qdrant_data():
    """Check raw points in Qdrant collection."""
    print("🔍 Debugging Qdrant data...")

    try:
        # 1. Collection info
        print("1️⃣ Getting collection info...")
        response = requests.get(
            "http://localhost:6333/collections/vietnamese_knowledge_base", timeout=5
        )
        print(f"   Collection response: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"📊 Collection Info:")
            print(f"   Status: {data['result']['status']}")
            print(f"   Points: {data['result']['points_count']}")
            print(f"   Vectors: {data['result']['vectors_count']}")
        else:
            print(f"   Error getting collection: {response.text}")
            return

        # 2. Raw points data
        print("\n2️⃣ Getting raw points...")
        scroll_response = requests.post(
            "http://localhost:6333/collections/vietnamese_knowledge_base/points/scroll",
            json={"limit": 10, "with_payload": True, "with_vector": False},
            timeout=5,
        )

        print(f"📋 Raw Points Data:")
        print(f"   Status: {scroll_response.status_code}")

        if scroll_response.status_code == 200:
            data = scroll_response.json()
            points = data.get("result", {}).get("points", [])
            print(f"   Points returned: {len(points)}")

            for i, point in enumerate(points[:3]):
                print(f"\n   Point {i+1}:")
                print(f"     ID: {point['id']}")

                payload = point.get("payload", {})
                print(f"     Payload keys: {list(payload.keys())}")

                if "page_content" in payload:
                    content = payload["page_content"]
                    print(f"     Content: {content[:100]}...")

                if "metadata" in payload:
                    metadata = payload["metadata"]
                    print(f"     Metadata: {metadata}")
        else:
            print(f"   Error getting points: {scroll_response.text}")

        # 3. Test direct embedding search với Qdrant API
        print(f"\n3️⃣ Testing direct Qdrant search...")

        # Generate test embedding (mock)
        test_vector = [0.1] * 768  # Mock E5-base vector

        search_response = requests.post(
            "http://localhost:6333/collections/vietnamese_knowledge_base/points/search",
            json={
                "vector": test_vector,
                "limit": 5,
                "with_payload": True,
                "score_threshold": 0.0,
            },
            timeout=5,
        )

        print(f"   Direct search status: {search_response.status_code}")
        if search_response.status_code == 200:
            results = search_response.json()
            points = results.get("result", [])
            print(f"   Direct search results: {len(points)}")

            for i, result in enumerate(points[:2]):
                score = result.get("score", 0)
                payload = result.get("payload", {})
                content = payload.get("page_content", "No content")
                print(
                    f"     Result {i+1}: score={score:.3f}, content={content[:50]}..."
                )
        else:
            print(f"   Error in direct search: {search_response.text}")

    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    check_qdrant_data()
