"""Test module for the RAG retriever."""

import pytest
from unittest.mock import Mock, patch
from langchain.schema import Document

from aiservice.src.rag.retriever import Retriever
from aiservice.src.rag.embedding import EmbeddingService, MultilinguaE5Embeddings
from aiservice.src.rag.vector_store import QdrantVectorService


@pytest.fixture
def mock_langchain_embeddings():
    embeddings = Mock(spec=MultilinguaE5Embeddings)
    return embeddings


@pytest.fixture
def mock_embedding_service(mock_langchain_embeddings):
    service = Mock(spec=EmbeddingService)
    service.get_langchain_embeddings.return_value = mock_langchain_embeddings
    return service


@pytest.fixture
def mock_vector_store():
    store = Mock(spec=QdrantVectorService)
    # Mock similarity_search_with_score to return List[Tuple[Document, float]]
    store.similarity_search_with_score.return_value = [
        (
            Document(
                page_content="Test document 1",
                metadata={"source": "test1.txt", "doc_id": "doc1"},
            ),
            0.95,
        ),
        (
            Document(
                page_content="Test document 2",
                metadata={"source": "test2.txt", "doc_id": "doc2"},
            ),
            0.85,
        ),
    ]
    return store


@pytest.mark.asyncio
class TestRetriever:
    """Test cases for the Retriever class."""

    async def test_retriever_initialization(
        self, mock_embedding_service, mock_vector_store
    ):
        """Test retriever initialization with and without parameters."""
        # Test with provided services
        retriever = Retriever(
            embedding_service=mock_embedding_service, vector_store=mock_vector_store
        )
        assert retriever.embedding_service == mock_embedding_service
        assert retriever.vector_store == mock_vector_store

        # Test default parameters
        assert retriever.config.top_k == 4
        assert retriever.config.score_threshold == 0.7

    async def test_get_relevant_documents(
        self, mock_embedding_service, mock_vector_store
    ):
        """Test retrieving relevant documents."""
        # Setup
        retriever = Retriever(
            embedding_service=mock_embedding_service,
            vector_store=mock_vector_store,
            top_k=2,
            score_threshold=0.8,
        )

        # Execute
        docs = await retriever._aget_relevant_documents(
            "test query", run_manager=Mock()
        )

        # Verify results
        assert len(docs) == 2
        assert isinstance(docs[0], Document)

        # Check first document
        assert docs[0].page_content == "Test document 1"
        assert docs[0].metadata["source"] == "test1.txt"
        assert docs[0].metadata["doc_id"] == "doc1"
        assert docs[0].metadata["score"] == 0.95

        # Check second document
        assert docs[1].page_content == "Test document 2"
        assert docs[1].metadata["source"] == "test2.txt"
        assert docs[1].metadata["doc_id"] == "doc2"
        assert docs[1].metadata["score"] == 0.85

        # Verify vector store was called correctly
        mock_vector_store.similarity_search_with_score.assert_called_once_with(
            query="test query", k=2, score_threshold=0.8
        )

    @patch("aiservice.src.rag.retriever.create_vietnamese_vector_store")
    async def test_retriever_with_none_vector_store(
        self, mock_create, mock_embedding_service
    ):
        """Test retriever handles None vector store correctly."""
        # Setup - create a mock store
        mock_store = Mock(spec=QdrantVectorService)
        mock_create.return_value = mock_store

        # Create retriever with None vector_store
        retriever = Retriever(
            embedding_service=mock_embedding_service, vector_store=None
        )

        # Verify vector store was created
        mock_create.assert_called_once()
        assert retriever.vector_store == mock_store

    async def test_retriever_error_handling(
        self, mock_embedding_service, mock_vector_store
    ):
        """Test error handling in retriever."""
        # Setup - make vector store raise an exception
        error_msg = "Test error"
        mock_vector_store.similarity_search_with_score.side_effect = Exception(
            error_msg
        )

        retriever = Retriever(
            embedding_service=mock_embedding_service, vector_store=mock_vector_store
        )

        # Verify exception is propagated
        with pytest.raises(Exception) as exc_info:
            await retriever._aget_relevant_documents("test query", run_manager=Mock())
        assert str(exc_info.value) == error_msg
