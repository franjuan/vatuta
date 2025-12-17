import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from src.rag.document_manager import DocumentManager


class TestAddDocuments(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp()
        with patch("src.rag.document_manager.HuggingFaceEmbeddings"):
            self.dm = DocumentManager(storage_dir=self.test_dir)
            self.dm.embeddings = MagicMock()
            # Fake embeddings for FAISS
            self.dm.embeddings.embed_documents.return_value = [[0.1] * 384]

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir)

    def test_add_documents_legacy(self) -> None:
        """Test the add_documents method which caused the original error."""
        doc = Document(page_content="Test content", metadata={"source": "test_source", "id": "test_id_1"})
        doc2 = Document(page_content="Test content 2", metadata={"source": "test_source"})

        with patch("src.rag.document_manager.FAISS") as MockFAISS:
            self.dm.vectorstore = MagicMock()
            MockFAISS.from_documents.return_value = self.dm.vectorstore

            success = self.dm.add_documents([doc, doc2])

            self.assertTrue(success)
            # Check if metadata was populated
            self.assertEqual(len(self.dm.documents_metadata), 2)

            # Check if IDs were generated/used correctly
            self.assertIn("test_id_1", self.dm.documents_metadata)
            # for doc2, it should be source_hash
            # We can iterate to find it
            keys = list(self.dm.documents_metadata.keys())
            self.assertTrue(any("test_source_" in k for k in keys if k != "test_id_1"))


if __name__ == "__main__":
    unittest.main()
