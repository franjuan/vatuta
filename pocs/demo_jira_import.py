"""
Demo script for JIRA import functionality

This script demonstrates how to use the JIRA importer and document manager.
"""

import os

from pocs.jira_importer import JIRAImporter
from src.models.config import ConfigLoader
from src.rag.qdrant_manager import QdrantDocumentManager


def demo_jira_import():
    """Demonstrate JIRA import functionality."""
    print("🚀 JIRA Import Demo")
    print("=" * 50)

    # Check if environment variables are set
    required_vars = ["JIRA_USER", "JIRA_API_TOKEN", "JIRA_INSTANCE_URL"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these variables in your environment or .env file")
        return

    try:
        # Initialize components
        print("🔧 Initializing components...")
        config = ConfigLoader.load("config/vatuta.yaml")
        jira_importer = JIRAImporter()
        doc_manager = QdrantDocumentManager(config.qdrant)

        # Show current stats
        stats = doc_manager.get_document_stats()
        print(f"📊 Current knowledge base: {stats['total_documents']} documents")

        # Import JIRA tickets
        print("\n📋 Importing JIRA tickets...")
        jira_docs = jira_importer.import_jira_tickets()

        # Import Confluence pages
        print("\n📄 Importing Confluence pages...")
        confluence_docs = jira_importer.import_confluence_pages()

        # Combine documents
        all_docs = jira_importer.create_document_list(jira_docs, confluence_docs)

        if all_docs:
            # Split documents
            print("\n✂️ Splitting documents...")
            final_docs = jira_importer.split_documents(all_docs)

            # Add to knowledge base
            print("\n💾 Adding to knowledge base...")
            success = doc_manager.add_documents(final_docs)

            if success:
                print("✅ Import completed successfully!")

                # Show updated stats
                new_stats = doc_manager.get_document_stats()
                print(f"📊 Updated knowledge base: {new_stats['total_documents']} documents")

                # Demo search
                print("\n🔍 Testing search functionality...")
                print("\n🔍 Testing search functionality...")
                search_results = doc_manager.vectorstore.similarity_search_with_score("epic requirements", k=3)
                if search_results:
                    print(f"Found {len(search_results)} relevant documents:")
                    for i, (doc, score) in enumerate(search_results, 1):
                        title = doc.metadata.get("summary", doc.metadata.get("title", "Untitled"))
                        print(f"  {i}. {title}")

                # Show recent documents
                print("\n📋 Recent documents:")
                recent_docs = doc_manager.list_documents(limit=5)
                for doc in recent_docs:
                    print(f"  - {doc['title']} ({doc['source']})")
            else:
                print("❌ Failed to add documents to knowledge base")
        else:
            print("⚠️ No documents to import")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    demo_jira_import()
