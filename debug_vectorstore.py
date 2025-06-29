# -*- coding: utf-8 -*-
"""Debug vectorstore direct methods."""

import sys

sys.path.append("src")

from src.rag.service import get_rag_service


async def test_vectorstore_direct():
    """Test direct vectorstore methods."""
    print("🔍 Testing direct vectorstore...")

    try:
        rag = get_rag_service()
        vectorstore = rag.vectorstore

        print(f"Collection: {vectorstore.collection_name}")

        # 1. Test similarity_search (no scores)
        print("\n1️⃣ Testing similarity_search (no scores)...")
        docs = vectorstore.similarity_search(query="Qdrant vector database", k=5)
        print(f"Found {len(docs)} documents")
        for i, doc in enumerate(docs[:2]):
            print(f"  Doc {i+1}: {doc.page_content[:60]}...")
            print(f"  Metadata: {doc.metadata}")

        # 2. Test similarity_search_with_score
        print("\n2️⃣ Testing similarity_search_with_score...")
        try:
            results = vectorstore.similarity_search_with_score(
                query="Qdrant vector database", k=5, score_threshold=0.0
            )
            print(f"Found {len(results)} results with scores")
            for i, (doc, score) in enumerate(results[:2]):
                print(f"  Result {i+1}: score={score:.3f}")
                print(f"    Content: {doc.page_content[:60]}...")
        except Exception as e:
            print(f"  Error in similarity_search_with_score: {e}")

        # 3. Test hybrid retriever initialization
        print("\n3️⃣ Testing hybrid retriever...")
        hybrid = rag.hybrid_retriever
        print(f"BM25 initialized: {hybrid.bm25_retriever is not None}")

        if hybrid.bm25_retriever:
            print(f"BM25 docs count: {len(hybrid.bm25_retriever.documents)}")
        else:
            print("BM25 not initialized - this could be the issue!")

            # Try to initialize BM25 với existing docs
            print("4️⃣ Trying to initialize BM25...")
            test_docs = vectorstore.similarity_search("test", k=10)
            if test_docs:
                hybrid.add_documents_to_bm25(test_docs)
                print(f"BM25 initialized with {len(test_docs)} docs")

        # 4. Test hybrid search after BM25 init
        print("\n5️⃣ Testing hybrid search...")
        try:
            hybrid_docs = await hybrid.hybrid_search(
                query="Qdrant vector database",
                k=5,
                method="embedding_only",  # Test embedding only first
            )
            print(f"Hybrid search found: {len(hybrid_docs)} docs")
            for i, doc in enumerate(hybrid_docs[:2]):
                score = doc.metadata.get("embedding_score", 0)
                print(f"  Doc {i+1}: score={score:.3f}")
                print(f"    Content: {doc.page_content[:60]}...")
        except Exception as e:
            print(f"  Error in hybrid search: {e}")
            import traceback

            traceback.print_exc()

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_vectorstore_direct())
