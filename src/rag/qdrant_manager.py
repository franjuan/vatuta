"""Qdrant-based Document Management System.

This module handles storing, retrieving, and managing documents using Qdrant vector database.
"""

import hashlib
import os
import uuid
from typing import Any, Dict, List, Optional, TypeAlias
from uuid import UUID

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    HasIdCondition,
    HasVectorCondition,
    IsEmptyCondition,
    IsNullCondition,
    MatchValue,
    NestedCondition,
    PointIdsList,
    VectorParams,
)

from src.models.config import QdrantConfig
from src.models.documents import ChunkRecord, DocumentUnit

QdrantFilterCondition: TypeAlias = (
    FieldCondition | IsEmptyCondition | IsNullCondition | HasIdCondition | HasVectorCondition | NestedCondition | Filter
)


class QdrantDocumentManager:
    """Manages documents in Qdrant vector database with LangChain integration."""

    def __init__(self, config: QdrantConfig) -> None:
        """Initialize Qdrant client and embeddings.

        Args:
            config: Qdrant configuration object
        """
        self.config = config

        # Get API key from environment variable (like other secrets)
        api_key = os.getenv("QDRANT_API_KEY")
        self.client = QdrantClient(url=config.url, api_key=api_key)
        self.collection_name = config.collection_name

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=config.embeddings_model)

        # Ensure collection exists
        self._ensure_collection()

        # Create LangChain vector store wrapper
        self.vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )

    def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name not in collection_names:
            # Get embedding dimension dynamically from the model
            # Create a sample embedding to determine vector size
            sample_vector = self.embeddings.embed_query("sample")
            vector_size = len(sample_vector)

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )
            print(f"✅ Created collection: {self.collection_name} (vector size: {vector_size})")
            print(f"✅ Created collection: {self.collection_name} (vector size: {vector_size})")

    def _generate_doc_id(self, doc: Document) -> str:
        """Generate a unique ID for a document based on content hash.

        Qdrant requires IDs to be UUIDs or unsigned integers.
        """
        # Use existing ID if available AND valid UUID
        if "id" in doc.metadata:
            try:
                # Validate if it's a UUID
                return str(uuid.UUID(str(doc.metadata["id"])))
            except ValueError:
                pass  # Not a valid UUID, generate one

        # Generate stable ID from content and source
        source = doc.metadata.get("source", "unknown")
        # Ensure content is string
        content = doc.page_content if doc.page_content else ""
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        unique_string = f"{source}_{content_hash}"

        # Generate deterministic UUID
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_string))

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

        print(f"📝 Adding {len(documents)} documents to Qdrant...")

        try:
            # Generate IDs for all documents
            doc_ids = [self._generate_doc_id(doc) for doc in documents]

            # Add documents to vector store
            # QdrantVectorStore handles upserts (overwriting existing IDs)
            self.vectorstore.add_documents(documents, ids=doc_ids)

            print(f"✅ Successfully added {len(documents)} documents")
            return True

        except Exception as e:
            print(f"❌ Error adding documents: {e}")
            import traceback

            traceback.print_exc()
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

            # Collect IDs from chunks (convert to UUIDs for Qdrant)
            chunk_ids = []
            for ch in chunks:
                if documents_by_id.get(ch.parent_document_id) is not None:
                    # Generate deterministic UUID from chunk_id
                    chunk_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(ch.chunk_id)))
                    chunk_ids.append(chunk_uuid)

            # Add to vector store
            print(f"🆕 Adding {len(lc_docs)} chunks to vector store...")
            self.vectorstore.add_documents(lc_docs, ids=chunk_ids)

            print(f"✅ Successfully added {len(lc_docs)} chunks")
            return True
        except Exception as e:
            print(f"❌ Error adding chunk records: {e}")
            import traceback

            traceback.print_exc()
            return False

    def delete_documents(
        self,
        source: Optional[str] = None,
        source_instance_id: Optional[str] = None,
    ) -> int:
        """Delete documents matching criteria from metadata.

        Args:
            source: Filter by source type (e.g., 'slack')
            source_instance_id: Filter by source instance ID (e.g., 'slack-main')

        Returns:
            Number of documents deleted.
        """
        if not source and not source_instance_id:
            print("⚠️ No deletion criteria specified")
            return 0

        try:
            # Build filter conditions
            conditions: List[QdrantFilterCondition] = []
            # Note: Qdrant vector store stores metadata nested under 'metadata' key
            if source:
                conditions.append(FieldCondition(key="metadata.source", match=MatchValue(value=source)))
            if source_instance_id:
                conditions.append(
                    FieldCondition(key="metadata.source_instance_id", match=MatchValue(value=source_instance_id))
                )

            # Create filter
            filter_query = Filter(must=conditions)

            # Get count before deletion (for reporting)
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_query,
                limit=10000,  # Get all matching
                with_payload=False,
                with_vectors=False,
            )
            count_before = len(scroll_result[0]) if scroll_result else 0

            # Delete with filter
            print(f"🗑️ Deleting {count_before} records...")
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=filter_query,
            )

            print(f"✅ Deleted {count_before} records")
            return count_before

        except Exception as e:
            print(f"❌ Error during deletion: {e}")
            import traceback

            traceback.print_exc()
            return 0

    def delete_documents_by_ids(self, ids: list[int | str | UUID]) -> bool:
        """Delete documents by their IDs.

        Args:
            ids: List of document/chunk IDs to delete

        Returns:
            True if successful, False otherwise
        """
        if not ids:
            return True

        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(points=ids),
            )
            print(f"✅ Deleted {len(ids)} documents by ID")
            return True
        except Exception as e:
            print(f"❌ Error deleting from vectorstore: {e}")
            return False

    def get_detailed_stats(self) -> Dict[str, Any]:
        """Get detailed statistics about documents and chunks by source.

        Returns:
            Dictionary containing:
            - total_documents: Total number of chunks/points
            - sources: Dict mapping source name to stats:
                - chunks_count: Number of chunks
                - documents_count: Number of unique parent documents
                - total_size: Total characters in content
                - mean_chunk_size: Average characters per chunk
                - mean_document_size: Average characters per document
                - mean_chunks_per_document: Average chunks per document
        """
        try:
            # Get collection info
            collection_info = self.client.get_collection(self.collection_name)
            total_points = collection_info.points_count

            # Aggregation storage
            # source -> { 'chunks': int, 'size': int, 'parent_ids': Set[str] }
            agg: Dict[str, Dict[str, Any]] = {}

            # Scroll through all points
            offset = None
            while True:
                scroll_result, offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )

                for point in scroll_result:
                    if point.payload:
                        # Extract metadata
                        metadata = point.payload.get("metadata", {})
                        if not isinstance(metadata, dict):
                            metadata = point.payload

                        # Get fields safely
                        raw_source = metadata.get("source") or point.payload.get("source")
                        source = str(raw_source) if raw_source else "unknown"

                        content = point.payload.get("page_content", "")
                        size = len(content) if content else 0

                        parent_id = metadata.get("parent_document_id") or metadata.get("doc_id") or "unknown"

                        # Initialize source stats if needed
                        if source not in agg:
                            agg[source] = {"chunks": 0, "size": 0, "parent_ids": set()}

                        # Update aggregates
                        agg[source]["chunks"] += 1
                        agg[source]["size"] += size
                        agg[source]["parent_ids"].add(parent_id)

                if offset is None:
                    break

            # Compute final stats
            sources_stats = {}
            for source, data in agg.items():
                chunks = data["chunks"]
                total_size = data["size"]
                num_docs = len(data["parent_ids"])

                sources_stats[source] = {
                    "chunks_count": chunks,
                    "documents_count": num_docs,
                    "total_size": total_size,
                    "mean_chunk_size": total_size / chunks if chunks > 0 else 0,
                    "mean_document_size": total_size / num_docs if num_docs > 0 else 0,
                    "mean_chunks_per_document": chunks / num_docs if num_docs > 0 else 0,
                }

            return {
                "total_documents": total_points,
                "sources": sources_stats,
                "storage_directory": f"Qdrant: {self.config.url}",
                "vectorstore_exists": True,
            }

        except Exception as e:
            print(f"⚠️ Error getting detailed stats: {e}")
            return {
                "total_documents": 0,
                "sources": {},
                "vectorstore_exists": False,
            }

    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about the documents in the database.

        Returns:
            Dictionary with document statistics
        """
        try:
            # Get collection info
            collection_info = self.client.get_collection(self.collection_name)
            total_docs = collection_info.points_count

            # Get sample of documents to count by source
            # Scroll through all points to get accurate source counts
            sources: Dict[str, int] = {}
            offset = None

            while True:
                scroll_result, offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )

                for point in scroll_result:
                    if point.payload:
                        # QdrantVectorStore nests metadata
                        metadata = point.payload.get("metadata", {})
                        # If metadata is not a dict (rare edge case), fallback
                        if not isinstance(metadata, dict):
                            metadata = point.payload

                        raw_source = metadata.get("source") or point.payload.get("source")
                        source = str(raw_source) if raw_source else "unknown"
                        sources[source] = sources.get(source, 0) + 1

                if offset is None:
                    break

            stats = {
                "total_documents": total_docs,
                "sources": sources,
                "storage_directory": f"Qdrant: {self.config.url}",
                "vectorstore_exists": True,
            }

            return stats

        except Exception as e:
            print(f"⚠️ Error getting stats: {e}")
            return {
                "total_documents": 0,
                "sources": {},
                "storage_directory": f"Qdrant: {self.config.url}",
                "vectorstore_exists": False,
            }

    def list_documents(
        self,
        source: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """List documents in the knowledge base.

        Args:
            source: Filter by source (jira, confluence, slack, etc.)
            limit: Maximum number of documents to return

        Returns:
            List of document information
        """
        try:
            # Build filter if source specified
            scroll_filter = None
            if source:
                scroll_filter = Filter(must=[FieldCondition(key="metadata.source", match=MatchValue(value=source))])

            # Scroll through collection
            scroll_result, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=scroll_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )

            documents = []
            for point in scroll_result:
                if point.payload:
                    metadata = point.payload.get("metadata", {})
                    # Fallback to top-level if needed
                    if not isinstance(metadata, dict):
                        metadata = point.payload

                    doc_info = {
                        "doc_id": str(point.id),
                        "title": metadata.get("title", point.payload.get("title", "Untitled")),
                        "source": metadata.get("source", point.payload.get("source", "unknown")),
                        "added_at": metadata.get("added_at", point.payload.get("added_at")),
                        "content_preview": metadata.get("content_preview", point.payload.get("content_preview", "")),
                    }
                    documents.append(doc_info)

            return documents

        except Exception as e:
            print(f"⚠️ Error listing documents: {e}")
            return []

    def clear_all_documents(self) -> bool:
        """Clear all documents from the knowledge base.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete the collection
            self.client.delete_collection(collection_name=self.collection_name)

            # Recreate it
            self._ensure_collection()

            # Recreate vector store wrapper
            self.vectorstore = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embeddings,
            )

            print("🗑️ Cleared all documents from knowledge base")
            return True

        except Exception as e:
            print(f"❌ Error clearing documents: {e}")
            return False

    def search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Any] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Document]:
        """Search for documents similar to the query with optional filtering.

        Args:
            query: The search query string.
            k: Number of documents to return.
            filter: Optional Qdrant filter (dict or Filter object).
            score_threshold: Minimum similarity score.

        Returns:
            List of matching LangChain Documents.
        """
        try:
            # We use similarity_search_with_score to apply threshold if needed
            docs_and_scores = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter,
            )

            results: List[Document] = []
            for doc, score in docs_and_scores:
                if score_threshold is not None and score < score_threshold:
                    continue
                # Add score to metadata for transparency
                doc.metadata["score"] = score
                results.append(doc)

            return results

        except Exception as e:
            print(f"❌ Error during search: {e}")
            return []

    def get_documents(self, filter: Any, limit: Optional[int] = None) -> List[Document]:
        """Retrieve documents matching a specific filter (exact match, no vector search).

        Args:
            filter: Qdrant filter (dict or Filter object).
            limit: Maximum number of documents to retrieve. If None, retrieves all (up to robust max).

        Returns:
            List of LangChain Documents.
        """
        try:
            # Use scroll to get points matching the filter
            # QdrantClient.scroll returns (points, next_page_offset)
            # If limit is None, we want "all" matching. Qdrant requires a limit for scroll.
            # We set a reasonably high limit (100) to cover most specific lookups.
            # If strictly "all" is needed, one would need to loop over offset.
            effective_limit = limit if limit is not None else 100

            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter,
                limit=effective_limit,
                with_payload=True,
                with_vectors=False,
            )

            documents: List[Document] = []
            for point in points:
                if not point.payload:
                    continue

                # Reconstruct LangChain Document
                content = point.payload.get("page_content", "")
                metadata = point.payload.get("metadata", {})
                if not isinstance(metadata, dict):
                    metadata = point.payload

                documents.append(Document(page_content=content, metadata=metadata))

            return documents

        except Exception as e:
            print(f"❌ Error during direct retrieval: {e}")
            return []


def main() -> None:
    """Demonstrate document management functionality."""
    from src.models.config import ConfigLoader

    print("📚 Qdrant Document Manager Demo")
    print("=" * 30)

    # Load configuration
    config = ConfigLoader.load("config/vatuta.yaml")

    # Initialize document manager
    manager = QdrantDocumentManager(config.qdrant)

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
