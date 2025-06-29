#!/usr/bin/env python3
"""
Test script để verify Qdrant integration và migration từ ChromaDB.
"""

import sys
import asyncio
import time
from pathlib import Path

# Add src to path
sys.path.append("src")


async def test_qdrant_connection():
    """Test kết nối đến Qdrant server."""
    print("🔌" + "=" * 50 + "🔌")
    print("🇻🇳  QDRANT CONNECTION TEST  🇻🇳")
    print("🔌" + "=" * 50 + "🔌")
    print()

    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url="http://localhost:6333")

        # Test connection
        collections = client.get_collections()
        print(f"✅ Qdrant server connected successfully!")
        print(f"📊 Existing collections: {len(collections.collections)}")

        if collections.collections:
            for collection in collections.collections:
                print(f"   📄 Collection: {collection.name}")

        return True

    except Exception as e:
        print(f"❌ Qdrant connection failed: {e}")
        print("💡 Hãy start Qdrant server với Docker:")
        print("   docker run -p 6333:6333 qdrant/qdrant")
        return False


async def test_qdrant_vectorstore():
    """Test QdrantVectorStore functionality."""
    print("\n🔧" + "=" * 50 + "🔧")
    print("🇻🇳  QDRANT VECTORSTORE TEST  🇻🇳")
    print("🔧" + "=" * 50 + "🔧")
    print()

    try:
        from rag.service import RAGService

        # Tạo RAGService với Qdrant
        print("📦 Creating RAGService với Qdrant vectorstore...")
        rag = RAGService(
            vector_store_type="qdrant",
            collection_name="test_vietnamese_kb",
        )

        print(f"✅ RAGService created with vector store type: {rag.vector_store_type}")

        # Test add some documents
        test_docs = [
            "Chỉ số huyết áp là thông số quan trọng cho sức khỏe.",
            "RAG (Retrieval-Augmented Generation) là công nghệ AI tiên tiến.",
            "Qdrant là vector database hiệu suất cao cho machine learning.",
        ]

        print("📄 Adding test documents...")
        for i, text in enumerate(test_docs):
            result = await rag.add_text(
                text=text, metadata={"source": f"test_doc_{i}", "type": "test"}
            )
            print(f"   ✅ Added doc {i+1}: {result['chunks_added']} chunks")

        # Test search
        print("\n🔍 Testing search functionality...")
        test_query = "huyết áp"

        search_result = await rag.search_only(question=test_query, k=3, method="hybrid")

        print(f"📊 Search results for '{test_query}':")
        print(f"   Documents found: {search_result['num_sources']}")
        print(f"   Confidence: {search_result['confidence_score']:.3f}")

        if search_result["sources"]:
            for i, source in enumerate(search_result["sources"][:2]):
                score = source["similarity_score"]
                print(f"   📄 Result {i+1}: score={score:.3f}")
                print(f"       Content: {source['content_preview'][:60]}...")

        return True

    except Exception as e:
        print(f"❌ QdrantVectorStore test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_migration_from_chroma():
    """Test migration từ ChromaDB sang Qdrant."""
    print("\n🔄" + "=" * 50 + "🔄")
    print("🇻🇳  MIGRATION FROM CHROMA TEST  🇻🇳")
    print("🔄" + "=" * 50 + "🔄")
    print()

    try:
        # Check if ChromaDB has data
        print("1️⃣ Checking existing ChromaDB data...")

        from rag.service import RAGService

        # Create ChromaDB service
        chroma_rag = RAGService(
            vector_store_type="chroma",
            collection_name="vietnamese_knowledge_base",
        )

        # Get ChromaDB collection info
        chroma_info = chroma_rag.vectorstore.get_collection_info()
        print(f"   📊 ChromaDB documents: {chroma_info.get('documents_count', 0)}")

        if chroma_info.get("documents_count", 0) == 0:
            print("   ⚠️  No data in ChromaDB to migrate")
            return True

        # Create Qdrant service for migration target
        print("\n2️⃣ Creating Qdrant service for migration...")
        qdrant_rag = RAGService(
            vector_store_type="qdrant",
            collection_name="migrated_vietnamese_kb",
        )

        # Perform migration
        print("\n3️⃣ Performing migration...")
        migration_result = (
            await qdrant_rag.vectorstore.qdrant_store.migrate_from_chroma(
                chroma_rag.vectorstore
            )
        )

        print(f"   ✅ Migration result: {migration_result['success']}")
        print(f"   📊 Documents migrated: {migration_result['migrated_count']}")
        print(f"   💬 Message: {migration_result['message']}")

        # Verify migration
        print("\n4️⃣ Verifying migration...")
        qdrant_info = qdrant_rag.vectorstore.get_collection_info()
        print(
            f"   📊 Qdrant documents after migration: {qdrant_info.get('vectors_count', 0)}"
        )

        # Test search on migrated data
        test_search = await qdrant_rag.search_only(
            question="RAG là gì", k=3, method="hybrid"
        )

        print(f"   🔍 Test search results: {test_search['num_sources']} documents")
        print(f"   📊 Confidence: {test_search['confidence_score']:.3f}")

        return migration_result["success"]

    except Exception as e:
        print(f"❌ Migration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_performance_comparison():
    """So sánh performance giữa ChromaDB và Qdrant."""
    print("\n⚡" + "=" * 50 + "⚡")
    print("🇻🇳  PERFORMANCE COMPARISON  🇻🇳")
    print("⚡" + "=" * 50 + "⚡")
    print()

    try:
        from rag.service import RAGService

        test_query = "Chỉ số huyết áp gần nhất trong ngày"
        test_runs = 3

        # Test ChromaDB
        print("1️⃣ Testing ChromaDB performance...")
        chroma_rag = RAGService(vector_store_type="chroma")

        chroma_times = []
        for i in range(test_runs):
            start_time = time.time()
            await chroma_rag.search_only(question=test_query, k=5, method="hybrid")
            end_time = time.time()
            chroma_times.append(end_time - start_time)

        chroma_avg = sum(chroma_times) / len(chroma_times)
        print(f"   ⏱️  ChromaDB average: {chroma_avg:.3f}s")

        # Test Qdrant
        print("\n2️⃣ Testing Qdrant performance...")
        qdrant_rag = RAGService(vector_store_type="qdrant")

        qdrant_times = []
        for i in range(test_runs):
            start_time = time.time()
            await qdrant_rag.search_only(question=test_query, k=5, method="hybrid")
            end_time = time.time()
            qdrant_times.append(end_time - start_time)

        qdrant_avg = sum(qdrant_times) / len(qdrant_times)
        print(f"   ⚡ Qdrant average: {qdrant_avg:.3f}s")

        # Comparison
        print(f"\n📊 PERFORMANCE COMPARISON:")
        print(f"   ChromaDB: {chroma_avg:.3f}s")
        print(f"   Qdrant:   {qdrant_avg:.3f}s")

        if qdrant_avg < chroma_avg:
            speedup = chroma_avg / qdrant_avg
            print(f"   🚀 Qdrant is {speedup:.2f}x faster!")
        else:
            slowdown = qdrant_avg / chroma_avg
            print(f"   📈 ChromaDB is {slowdown:.2f}x faster")

        return True

    except Exception as e:
        print(f"❌ Performance comparison failed: {e}")
        return False


async def main():
    """Main test function."""
    print("🎯" + "=" * 60 + "🎯")
    print("🇻🇳  QDRANT INTEGRATION COMPREHENSIVE TEST  🇻🇳")
    print("🎯" + "=" * 60 + "🎯")
    print()

    tests = [
        ("Qdrant Connection", test_qdrant_connection),
        ("Qdrant VectorStore", test_qdrant_vectorstore),
        ("Migration from ChromaDB", test_migration_from_chroma),
        ("Performance Comparison", test_performance_comparison),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            print(f"\n🔍 Running: {test_name}")
            result = await test_func()
            results[test_name] = result
            print(
                f"{'✅' if result else '❌'} {test_name}: {'PASSED' if result else 'FAILED'}"
            )
        except Exception as e:
            print(f"❌ {test_name}: FAILED with exception: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY:")
    print("=" * 60)

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name}: {status}")

    print(f"\n🎯 Overall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED! Qdrant integration is working correctly!")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")

    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
