"""
Punto de entrada principal para Vatuta
"""

from pocs.bedrock_poc import main as bedrock_main
from pocs.gitlab_importer import GitLabImporter
from pocs.jira_importer import JIRAImporter
from pocs.slack_importer import SlackImporter
from src.rag.document_manager import DocumentManager


def main():
    """
    Función principal de Vatuta
    """
    print("🤖 Vatuta - Personal AI Assistant")
    print("=" * 40)

    # Initialize document manager
    doc_manager = DocumentManager()

    # Show current document stats
    stats = doc_manager.get_document_stats()
    print(f"📊 Knowledge Base Stats: {stats['total_documents']} documents")

    if stats["total_documents"] == 0:
        print("\n🔄 No documents found. Starting JIRA import process...")

        try:
            # Initialize JIRA importer
            jira_importer = JIRAImporter()

            # Import JIRA tickets
            print("\n📋 Importing JIRA tickets...")
            jira_docs = jira_importer.import_jira_tickets()

            # Import Confluence pages
            print("\n📄 Importing Confluence pages...")
            confluence_docs = jira_importer.import_confluence_pages()

            # Import Slack messages (if configured)
            print("\n💬 Importing Slack conversations...")
            slack_docs = SlackImporter().import_slack()

            # Import GitLab items (if configured)
            print("\n🦊 Importing GitLab issues/MRs...")
            try:
                gitlab_docs = GitLabImporter().import_gitlab(include="both", limit_per_project=50)
            except Exception as _e:
                print("⚠️ GitLab not configured or import failed; skipping GitLab import")
                gitlab_docs = []

            # Combine all documents
            all_docs = jira_importer.create_document_list(jira_docs, confluence_docs) + slack_docs + gitlab_docs

            # Split documents if needed
            final_docs = jira_importer.split_documents(all_docs)

            # Add to document manager
            if final_docs:
                success = doc_manager.add_documents(final_docs)
                if success:
                    print("✅ Documents successfully added to knowledge base!")
                else:
                    print("❌ Failed to add documents to knowledge base")
            else:
                print("⚠️ No documents to add")

        except Exception as e:
            print(f"❌ Error during JIRA import: {e}")
            print("Continuing with existing documents...")

    # Show recent documents
    recent_docs = doc_manager.list_documents(limit=5)
    if recent_docs:
        print(f"\n📋 Recent documents ({len(recent_docs)}):")
        for doc in recent_docs:
            print(f"  - {doc['title']} ({doc['source']})")

    # Demo search functionality
    print("\n🔍 Testing search functionality...")
    search_results = doc_manager.search_documents("epic requirements", k=3)
    if search_results:
        print(f"Found {len(search_results)} relevant documents")
        for i, result in enumerate(search_results, 1):
            title = result["metadata"].get("summary", result["metadata"].get("title", "Untitled"))
            print(f"  {i}. {title}")

    # Ejecutar PoC de Bedrock
    print("\n🚀 Starting Bedrock PoC...")
    bedrock_main()


if __name__ == "__main__":
    main()
