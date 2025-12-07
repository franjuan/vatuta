"""Document Management System.

This module handles storing, retrieving, and managing documents in the knowledge base.
"""

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

    def add_documents(self, documents: List[Document]) -> bool:
        """
        Add documents to the knowledge base.

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

            # Create or update vector store
            if self.vectorstore is None:
                print("🆕 Creating new vector store from chunks...")
                self.vectorstore = FAISS.from_documents(lc_docs, self.embeddings)
            else:
                print("🔄 Updating existing vector store with chunks...")
                new_vectorstore = FAISS.from_documents(lc_docs, self.embeddings)
                self.vectorstore.merge_from(new_vectorstore)

            # Update per-document metadata summary
            now_iso = datetime.now().isoformat()
            for ch in chunks:
                parent = documents_by_id.get(ch.parent_document_id)
                if parent is None:
                    continue
                key = parent.document_id
                # Keep a short, stable summary per source document
                self.documents_metadata[key] = {
                    "document_id": parent.document_id,
                    "source": parent.source,
                    "source_instance_id": parent.source_instance_id,
                    "source_doc_id": parent.source_doc_id,
                    "title": parent.title or "Untitled",
                    "uri": parent.uri,
                    "added_at": now_iso,
                }

            self._save_metadata()
            self._save_vectorstore()

            print(f"✅ Successfully added {len(lc_docs)} chunks")
            return True
        except Exception as e:
            print(f"❌ Error adding chunk records: {e}")
            return False

    def _generate_doc_id(self, doc: Document) -> str:
        """Generate a unique ID for a document."""
        if "issue_id" in doc.metadata:
            return f"jira_{doc.metadata['issue_id']}"
        elif "page_id" in doc.metadata:
            return f"confluence_{doc.metadata['page_id']}"
        elif "gitlab_issue_iid" in doc.metadata:
            project = doc.metadata.get("gitlab_project_id", "proj")
            return f"gitlab_issue_{project}_{doc.metadata['gitlab_issue_iid']}"
        elif "gitlab_mr_iid" in doc.metadata:
            project = doc.metadata.get("gitlab_project_id", "proj")
            return f"gitlab_mr_{project}_{doc.metadata['gitlab_mr_iid']}"
        else:
            # Fallback to content hash
            import hashlib

            content_hash = hashlib.md5(doc.page_content.encode(), usedforsecurity=False).hexdigest()[:8]
            return f"doc_{content_hash}"

    def search_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for documents using semantic similarity.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of search results with document content and metadata
        """
        if self.vectorstore is None:
            print("⚠️ No documents in knowledge base")
            return []

        try:
            # Perform similarity search
            docs = self.vectorstore.similarity_search(query, k=k)

            results = []
            for doc in docs:
                doc_id = self._generate_doc_id(doc)
                result = {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "doc_id": doc_id,
                    "stored_metadata": self.documents_metadata.get(doc_id, {}),
                }
                results.append(result)

            print(f"🔍 Found {len(results)} relevant documents for query: '{query}'")
            return results

        except Exception as e:
            print(f"❌ Error searching documents: {e}")
            return []

    def get_document_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the documents in the knowledge base.

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
        """
        List documents in the knowledge base.

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
        """
        Clear all documents from the knowledge base.

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
