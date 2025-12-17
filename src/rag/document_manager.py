"""Document Management System.

This module handles storing, retrieving, and managing documents in the knowledge base.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.models.documents import ChunkRecord, DocumentUnit


class DocumentManager:
    """Manages documents in the knowledge base with vector storage and persistence."""

    def __init__(
        self,
        storage_dir: str = "data",
        embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        """Initialize application state.

        Args:
            storage_dir: Directory to store documents and vector database
            embeddings_model: HuggingFace model for embeddings
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)

        self.embeddings_model = embeddings_model
        self.embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)

        self.vectorstore: Optional[FAISS] = None
        self.documents_metadata_file = self.storage_dir / "documents_metadata.json"
        self.vectorstore_dir = self.storage_dir / "vectorstore"

        # Load existing data if available
        self._load_existing_data()

    def _load_existing_data(self) -> None:
        """Load existing documents metadata and vector store."""
        try:
            if self.documents_metadata_file.exists():
                with open(self.documents_metadata_file, "r") as f:
                    self.documents_metadata = json.load(f)
            else:
                self.documents_metadata = {}

            if self.vectorstore_dir.exists():
                self.vectorstore = FAISS.load_local(
                    str(self.vectorstore_dir),
                    self.embeddings,
                    allow_dangerous_deserialization=True,  # We control the data source
                )
                print(f"📚 Loaded existing vector store with {len(self.documents_metadata)} documents")
            else:
                print("📚 No existing vector store found. Starting fresh.")

        except Exception as e:
            print(f"⚠️ Error loading existing data: {e}")
            self.documents_metadata = {}
            self.vectorstore = None

    def _save_metadata(self) -> None:
        """Save documents metadata to file."""
        try:
            with open(self.documents_metadata_file, "w") as f:
                json.dump(self.documents_metadata, f, indent=2)
        except Exception as e:
            print(f"⚠️ Error saving metadata: {e}")

    def _save_vectorstore(self) -> None:
        """Save vector store to file."""
        try:
            if self.vectorstore:
                self.vectorstore.save_local(str(self.vectorstore_dir))
        except Exception as e:
            print(f"⚠️ Error saving vector store: {e}")

    def _generate_doc_id(self, doc: Document) -> str:
        """Generate a unique ID for a document based on content hash."""
        # Use existing ID if available
        if "id" in doc.metadata:
            return str(doc.metadata["id"])

        # Generate stable ID from content and source
        source = doc.metadata.get("source", "unknown")
        content_hash = hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()
        return f"{source}_{content_hash}"

    def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to the knowledge base.

        Args:
            documents: List of Document objects to add

        Returns:
            True if successful, False otherwise
        """
        if not documents:
            print("⚠️ No documents to add")
            return False

        print(f"📝 Adding {len(documents)} documents to knowledge base...")

        try:
            # Create or update vector store
            if self.vectorstore is None:
                print("🆕 Creating new vector store...")
                self.vectorstore = FAISS.from_documents(documents, self.embeddings)
            else:
                print("🔄 Updating existing vector store...")
                # Add new documents to existing vector store
                new_vectorstore = FAISS.from_documents(documents, self.embeddings)
                self.vectorstore.merge_from(new_vectorstore)

            # Update metadata
            for doc in documents:
                doc_id = self._generate_doc_id(doc)
                self.documents_metadata[doc_id] = {
                    "source": doc.metadata.get("source", "unknown"),
                    "issue_id": doc.metadata.get("issue_id"),
                    "page_id": doc.metadata.get("page_id"),
                    "title": doc.metadata.get("title", doc.metadata.get("summary", "Untitled")),
                    "added_at": datetime.now().isoformat(),
                    "content_preview": (
                        doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    ),
                }

            # Save data
            self._save_metadata()
            self._save_vectorstore()

            print(f"✅ Successfully added {len(documents)} documents")
            return True

        except Exception as e:
            print(f"❌ Error adding documents: {e}")
            return False

    def add_chunk_records(
        self,
        chunks: List[ChunkRecord],
        documents_by_id: Dict[str, DocumentUnit],
    ) -> bool:
        """Add ChunkRecord items by converting them to LangChain Documents.

        Args:
            chunks: List of ChunkRecord instances to add
            documents_by_id: Mapping from parent_document_id to DocumentUnit

        Returns:
            True if successful, False otherwise
        """
        try:

            if not chunks:
                print("⚠️ No chunk records to add")
                return False

            lc_docs: List[Document] = []
            for ch in chunks:
                parent = documents_by_id.get(ch.parent_document_id)
                if parent is None:
                    print(f"⚠️ Skipping chunk {ch.chunk_id}: missing parent document {ch.parent_document_id}")
                    continue
                lc_docs.append(ch.to_langchain_document(parent))

            if not lc_docs:
                print("⚠️ No valid chunks to add after parent resolution")
                return False

            # Collect IDs from chunks
            chunk_ids = [ch.chunk_id for ch in chunks]

            # Create or update vector store
            print(f"🆕 Adding {len(lc_docs)} chunks to vector store...")
            if self.vectorstore is None:
                self.vectorstore = FAISS.from_documents(lc_docs, self.embeddings, ids=chunk_ids)
            else:
                self.vectorstore.add_documents(lc_docs, ids=chunk_ids)

            # Update per-document metadata summary
            now_iso = datetime.now().isoformat()

            # Use chunk_id as the key in metadata to match vectorstore IDs
            for ch in chunks:
                parent = documents_by_id.get(ch.parent_document_id)
                if parent is None:
                    continue

                # store metadata keyed by chunk_id so we can delete it later
                self.documents_metadata[ch.chunk_id] = {
                    "document_id": parent.document_id,  # Link to parent
                    "chunk_id": ch.chunk_id,
                    "source": parent.source,
                    "source_instance_id": parent.source_instance_id,
                    "source_doc_id": parent.source_doc_id,  # e.g. thread_ts
                    "title": parent.title or "Untitled",
                    "uri": parent.uri,
                    "added_at": now_iso,
                    "content_preview": ch.text[:100],
                }

            self._save_metadata()
            self._save_vectorstore()

            print(f"✅ Successfully added {len(lc_docs)} chunks")
            return True
        except Exception as e:
            print(f"❌ Error adding chunk records: {e}")
            import traceback

            traceback.print_exc()
            return False

    def delete_documents(self, source: Optional[str] = None, source_instance_id: Optional[str] = None) -> int:
        """Delete documents matching criteria from metadata and vectorstore.

        Args:
            source: Filter by source type (e.g., 'slack')
            source_instance_id: Filter by source instance ID (e.g., 'slack-main')

        Returns:
            Number of documents deleted.
        """
        # Find ID keys in metadata to delete
        ids_to_delete = []
        for key, meta in self.documents_metadata.items():
            if source and meta.get("source") != source:
                continue
            if source_instance_id and meta.get("source_instance_id") != source_instance_id:
                continue
            ids_to_delete.append(key)

        if not ids_to_delete:
            print("⚠️ No documents matched deletion criteria")
            return 0

        print(f"🗑️ Deleting {len(ids_to_delete)} records...")

        try:
            # 1. Delete from VectorStore
            if self.vectorstore:
                try:
                    self.vectorstore.delete(ids_to_delete)
                except Exception as e:
                    print(f"⚠️ FAISS deletion error (possibly IDs not found): {e}")

            # 2. Delete from Metadata
            for key in ids_to_delete:
                self.documents_metadata.pop(key, None)

            self._save_metadata()
            self._save_vectorstore()

            print(f"✅ Deleted {len(ids_to_delete)} records")
            return len(ids_to_delete)

        except Exception as e:
            print(f"❌ Error during deletion: {e}")
            import traceback

            traceback.print_exc()
            return 0

    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about the documents in the knowledge base.

        Returns:
            Dictionary with document statistics
        """
        total_docs = len(self.documents_metadata)

        # Count by source
        sources: Dict[str, int] = {}
        for metadata in self.documents_metadata.values():
            source = metadata.get("source", "unknown")
            sources[source] = sources.get(source, 0) + 1

        stats = {
            "total_documents": total_docs,
            "sources": sources,
            "storage_directory": str(self.storage_dir),
            "vectorstore_exists": self.vectorstore is not None,
        }

        return stats

    def list_documents(self, source: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """List documents in the knowledge base.

        Args:
            source: Filter by source (jira, confluence, etc.)
            limit: Maximum number of documents to return

        Returns:
            List of document information
        """
        documents = []

        for doc_id, metadata in self.documents_metadata.items():
            if source and metadata.get("source") != source:
                continue

            doc_info = {
                "doc_id": doc_id,
                "title": metadata.get("title", "Untitled"),
                "source": metadata.get("source", "unknown"),
                "added_at": metadata.get("added_at"),
                "content_preview": metadata.get("content_preview", ""),
            }
            documents.append(doc_info)

        # Sort by added_at (newest first)
        documents.sort(key=lambda x: x.get("added_at", ""), reverse=True)

        return documents[:limit]

    def clear_all_documents(self) -> bool:
        """Clear all documents from the knowledge base.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Clear in-memory data
            self.vectorstore = None
            self.documents_metadata = {}

            # Remove files
            if self.documents_metadata_file.exists():
                self.documents_metadata_file.unlink()

            if self.vectorstore_dir.exists():
                import shutil

                shutil.rmtree(self.vectorstore_dir)

            print("🗑️ Cleared all documents from knowledge base")
            return True

        except Exception as e:
            print(f"❌ Error clearing documents: {e}")
            return False

    def delete_documents_by_ids(self, ids: List[str]) -> bool:
        """Delete documents by their IDs."""
        try:
            # Delete from metadata
            for i in ids:
                self.documents_metadata.pop(i, None)
            self._save_metadata()

            # Delete from vectorstore
            if self.vectorstore:
                self.vectorstore.delete(ids)
                self._save_vectorstore()
            return True
        except Exception as e:
            print(f"❌ Error deleting from vectorstore: {e}")
            return False


def main() -> None:
    """Demonstrate document management functionality."""
    print("📚 Document Manager Demo")
    print("=" * 30)

    # Initialize document manager
    manager = DocumentManager()

    # Show current stats
    stats = manager.get_document_stats()
    print(f"📊 Current stats: {stats}")

    # List recent documents
    recent_docs = manager.list_documents(limit=5)
    print(f"\n📋 Recent documents ({len(recent_docs)}):")
    for doc in recent_docs:
        print(f"  - {doc['title']} ({doc['source']}) - {doc['added_at']}")


if __name__ == "__main__":
    main()
