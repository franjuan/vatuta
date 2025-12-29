"""Tests for DocumentManager deletion functionality."""

import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from src.models.documents import ChunkRecord, DocumentUnit
from src.rag.document_manager import DocumentManager


class TestDocumentManagerDeletion(unittest.TestCase):
    """Test cases for DocumentManager deletion."""

    def setUp(self) -> None:
        """Set up temporary directory and instance."""
        self.test_dir = tempfile.mkdtemp()

        # Mock embeddings to avoid heavy loading
        with patch("src.rag.document_manager.HuggingFaceEmbeddings"):
            self.dm = DocumentManager(storage_dir=self.test_dir)
            # Mock the internal embeddings object used by FAISS
            self.dm.embeddings = MagicMock()
            self.dm.embeddings.embed_documents.return_value = [[0.1] * 384] * 10
            self.dm.embeddings.embed_query.return_value = [0.1] * 384

    def tearDown(self) -> None:
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir)

    def test_add_and_delete_documents(self) -> None:
        """Test adding chunks and then deleting them by source."""
        # Create dummy document and chunks
        doc = DocumentUnit(
            document_id="doc1",
            source="slack",
            source_instance_id="slack-main",
            source_doc_id="thread1",
            title="Thread 1",
        )
        chunks = [
            ChunkRecord(
                chunk_id="c1",
                parent_document_id="doc1",
                parent_chunk_id=None,
                level=0,
                chunk_index=0,
                text="Hello",
                system_tags=[],
                embedding_model="test",
                chunking_strategy="test",
                content_hash="hash1",
            ),
            ChunkRecord(
                chunk_id="c2",
                parent_document_id="doc1",
                parent_chunk_id=None,
                level=0,
                chunk_index=1,
                text="World",
                system_tags=[],
                embedding_model="test",
                chunking_strategy="test",
                content_hash="hash2",
            ),
        ]

        docs_by_id = {"doc1": doc}

        # Mock FAISS
        with patch("src.rag.document_manager.FAISS") as MockFAISS:
            mock_vs = MagicMock()
            MockFAISS.from_documents.return_value = mock_vs
            self.dm.vectorstore = mock_vs

            # ADD
            success = self.dm.add_chunk_records(chunks, docs_by_id)
            self.assertTrue(success)
            self.assertEqual(len(self.dm.documents_metadata), 2)

            # DELETE Slack
            count = self.dm.delete_documents(source="slack")
            self.assertEqual(count, 2)
            self.assertEqual(len(self.dm.documents_metadata), 0)
            mock_vs.delete.assert_called()

    def test_filtered_deletion(self) -> None:
        """Test deleting only specific source."""
        # Doc 1: Slack
        doc1 = DocumentUnit(
            document_id="d1", source="slack", source_instance_id="s1", source_doc_id="ref1", title="Slack Doc"
        )
        c1 = ChunkRecord(
            chunk_id="c1",
            parent_document_id="d1",
            parent_chunk_id=None,
            level=0,
            chunk_index=0,
            text="txt",
            system_tags=[],
            content_hash="hash1",
        )

        # Doc 2: Jira
        doc2 = DocumentUnit(
            document_id="d2", source="jira", source_instance_id="j1", source_doc_id="ref2", title="Jira Doc"
        )
        c2 = ChunkRecord(
            chunk_id="c2",
            parent_document_id="d2",
            parent_chunk_id=None,
            level=0,
            chunk_index=0,
            text="txt",
            system_tags=[],
            content_hash="hash2",
        )

        docs_by_id = {"d1": doc1, "d2": doc2}

        with patch("src.rag.document_manager.FAISS"):
            self.dm.vectorstore = MagicMock()
            self.dm.add_chunk_records([c1, c2], docs_by_id)

            # Delete Jira only
            deleted = self.dm.delete_documents(source="jira")
            self.assertEqual(deleted, 1)

            # Verify Slack remains
            self.assertIn("c1", self.dm.documents_metadata)
            self.assertNotIn("c2", self.dm.documents_metadata)


if __name__ == "__main__":
    unittest.main()
