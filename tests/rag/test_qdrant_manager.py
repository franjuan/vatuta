import uuid
from typing import List
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_qdrant.qdrant import QdrantVectorStore
from qdrant_client.qdrant_client import QdrantClient

from src.models.config import QdrantConfig
from src.models.documents import ChunkRecord, DocumentUnit
from src.rag.qdrant_manager import QdrantDocumentManager


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 384 for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * 384


@pytest.fixture
def mock_qdrant_config() -> QdrantConfig:
    return QdrantConfig(url="http://localhost:6333", collection_name="vatuta_test", embeddings_model="all-MiniLM-L6-v2")


@pytest.fixture
def mock_qdrant_client() -> QdrantClient:
    with patch("src.rag.qdrant_manager.QdrantClient") as mock:
        yield mock


@pytest.fixture
def mock_embedding_function() -> HuggingFaceEmbeddings:
    with patch("src.rag.qdrant_manager.HuggingFaceEmbeddings") as mock:
        mock.return_value = FakeEmbeddings()
        yield mock


@pytest.fixture
def mock_qdrant_vectorstore() -> QdrantVectorStore:
    with patch("src.rag.qdrant_manager.QdrantVectorStore") as mock:
        yield mock


@pytest.fixture
def manager(
    mock_qdrant_config: QdrantConfig,
    mock_qdrant_client: QdrantClient,
    mock_embedding_function: HuggingFaceEmbeddings,
    mock_qdrant_vectorstore: QdrantVectorStore,
) -> QdrantDocumentManager:
    return QdrantDocumentManager(mock_qdrant_config)


def test_init(
    manager: QdrantDocumentManager,
    mock_qdrant_client: QdrantClient,
    mock_embedding_function: HuggingFaceEmbeddings,
    mock_qdrant_vectorstore: QdrantVectorStore,
) -> None:
    assert manager.collection_name == "vatuta_test"
    mock_qdrant_client.return_value.get_collections.assert_called_once()
    mock_embedding_function.assert_called_once()
    mock_qdrant_vectorstore.assert_called_once()


def test_generate_doc_id(manager: QdrantDocumentManager) -> None:
    valid_uuid = "123e4567-e89b-12d3-a456-426614174000"
    doc = Document(page_content="test content", metadata={"source": "test", "id": valid_uuid})
    assert manager._generate_doc_id(doc) == valid_uuid

    doc_no_id = Document(page_content="test content", metadata={"source": "test"})
    doc_no_id = Document(page_content="test content", metadata={"source": "test"})
    # Should be deterministic
    id1 = manager._generate_doc_id(doc_no_id)
    id2 = manager._generate_doc_id(doc_no_id)
    assert id1 == id2

    # Check if it's a valid UUID

    try:
        uuid.UUID(id1)
    except ValueError:
        pytest.fail("Generated ID is not a valid UUID")


def test_add_documents(manager: QdrantDocumentManager) -> None:
    # vectorstore is already mocked by the class patch, and assigned to manager.vectorstore

    docs = [
        Document(page_content="test 1", metadata={"source": "source1"}),
        Document(page_content="test 2", metadata={"source": "source2"}),
    ]

    success = manager.add_documents(docs)

    assert success is True
    manager.vectorstore.add_documents.assert_called_once()

    # Check that IDs were passed to add_documents
    call_args = manager.vectorstore.add_documents.call_args
    assert "ids" in call_args[1]
    assert len(call_args[1]["ids"]) == 2
    # Verify IDs are generated correctly (valid UUIDs)
    try:
        uuid.UUID(str(call_args[1]["ids"][0]))
    except ValueError:
        pytest.fail("ID passed to add_documents is not a valid UUID")


def test_add_chunk_records(manager: QdrantDocumentManager) -> None:
    # vectorstore is mocked class instance

    doc_unit = DocumentUnit(
        document_id="doc1",
        source="test",
        source_doc_id="s1",
        source_instance_id="inst1",
        uri="http://example.com",
        title="Test Doc",
        content_hash="hash",
    )

    chunks = [
        ChunkRecord(
            chunk_id="chunk1", parent_document_id="doc1", chunk_index=0, text="chunk content", system_tags=["tag1"]
        )
    ]

    docs_by_id = {"doc1": doc_unit}

    success = manager.add_chunk_records(chunks, docs_by_id)

    assert success is True
    manager.vectorstore.add_documents.assert_called_once()

    # Check payload construction
    call_args = manager.vectorstore.add_documents.call_args
    added_docs = call_args[0][0]
    first_doc = added_docs[0]

    assert first_doc.page_content == "chunk content"
    assert first_doc.metadata["document_id"] == "doc1"
    assert first_doc.metadata["title"] == "Test Doc"
    assert first_doc.metadata["chunk_id"] == "chunk1"
    assert "tag1" in first_doc.metadata["system_tags"]

    # Verify generated IDs are valid UUIDs
    call_args = manager.vectorstore.add_documents.call_args
    passed_ids = call_args[1]["ids"]
    assert len(passed_ids) == 1
    import uuid

    try:
        uuid.UUID(passed_ids[0])
    except ValueError:
        pytest.fail("Chunk ID passed to add_documents is not a valid UUID")


def test_delete_documents_by_filter(manager: QdrantDocumentManager) -> None:
    manager.client = MagicMock()
    # Mock scroll response for the count check before deletion
    # scroll returns (points, offset)
    manager.client.scroll.return_value = ([MagicMock()] * 5, None)

    deleted_count = manager.delete_documents(source="test_source")

    assert deleted_count == 5
    manager.client.delete.assert_called_once()

    # Check filter
    call_args = manager.client.delete.call_args
    points_selector = call_args[1]["points_selector"]
    # Verify it constructs a FilterSelector (or just Filter object)
    # Since we passed a Filter object directly to delete, points_selector IS the Filter.
    assert points_selector.must[0].key == "metadata.source"
    assert points_selector.must[0].match.value == "test_source"


def test_delete_documents_by_ids(manager: QdrantDocumentManager) -> None:
    manager.client = MagicMock()

    ids: List[int | str | UUID] = ["id1", "id2"]
    success = manager.delete_documents_by_ids(ids)

    assert success is True
    manager.client.delete.assert_called_once()
    call_args = manager.client.delete.call_args
    points_selector = call_args[1]["points_selector"]
    # Verify it uses PointIdsList
    assert points_selector.points == ids


def test_get_document_stats(manager: QdrantDocumentManager) -> None:
    manager.client = MagicMock()

    # Mock collection info using MagicMock to avoid Pydantic validation issues
    collection_info = MagicMock()
    collection_info.points_count = 100
    manager.client.get_collection.return_value = collection_info

    # Mock scroll for source counting
    # Return empty list to simulate no docs found during source count scan, or some docs
    manager.client.scroll.return_value = ([], None)

    stats = manager.get_document_stats()

    assert stats["total_documents"] == 100
    assert "sources" in stats
