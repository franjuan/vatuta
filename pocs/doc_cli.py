"""
Document CLI Interface

Command-line interface for importing documents from JIRA, Confluence, and Slack,
and for searching/listing the local knowledge base.
"""

import argparse
import sys

from pocs.gitlab_importer import GitLabImporter
from pocs.jira_importer import JIRAImporter
from pocs.rag_poc import main as rag_main
from pocs.slack_importer import SlackImporter
from pocs.tools_example import main as tools_main
from src.models.config import ConfigLoader
from src.rag.qdrant_manager import QdrantDocumentManager


def import_jira_tickets(args):
    """Import JIRA tickets with custom JQL query."""
    try:
        importer = JIRAImporter()
        docs = importer.import_jira_tickets(args.jql, args.max_results)

        if docs:
            config = ConfigLoader.load("config/vatuta.yaml")
            doc_manager = QdrantDocumentManager(config.qdrant)
            success = doc_manager.add_documents(docs)
            if success:
                print(f"✅ Successfully imported {len(docs)} JIRA tickets")
            else:
                print("❌ Failed to save documents")
        else:
            print("⚠️ No tickets found")

    except Exception as e:
        print(f"❌ Error importing JIRA tickets: {e}")


def import_confluence_pages(args):
    """Import Confluence pages."""
    try:
        importer = JIRAImporter()
        docs = importer.import_confluence_pages(args.space_key, args.limit)

        if docs:
            config = ConfigLoader.load("config/vatuta.yaml")
            doc_manager = QdrantDocumentManager(config.qdrant)
            success = doc_manager.add_documents(docs)
            if success:
                print(f"✅ Successfully imported {len(docs)} Confluence pages")
            else:
                print("❌ Failed to save documents")
        else:
            print("⚠️ No pages found")

    except Exception as e:
        print(f"❌ Error importing Confluence pages: {e}")


def search_documents(args):
    """Search documents in the knowledge base."""
    try:
        config = ConfigLoader.load("config/vatuta.yaml")
        doc_manager = QdrantDocumentManager(config.qdrant)
        results = doc_manager.vectorstore.similarity_search_with_score(args.query, k=args.k)

        # Format results to match previous interface
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "metadata": doc.metadata,
                "content": doc.page_content,
                "score": score
            })

        if formatted_results:
            print(f"🔍 Found {len(formatted_results)} relevant documents:")
            for i, result in enumerate(formatted_results, 1):
                title = result["metadata"].get("summary", result["metadata"].get("title", "Untitled"))
                source = result["metadata"].get("source", "unknown")
                print(f"  {i}. {title} ({source})")
                print(f"     Content preview: {result['content'][:100]}...")
                print()
        else:
            print("No relevant documents found")

    except Exception as e:
        print(f"❌ Error searching documents: {e}")


def list_documents(args):
    """List documents in the knowledge base."""
    try:
        config = ConfigLoader.load("config/vatuta.yaml")
        doc_manager = QdrantDocumentManager(config.qdrant)
        docs = doc_manager.list_documents(args.source, args.limit)

        if docs:
            print(f"📋 Documents ({len(docs)}):")
            for doc in docs:
                print(f"  - {doc['title']} ({doc['source']}) - {doc['added_at']}")
        else:
            print("No documents found")

    except Exception as e:
        print(f"❌ Error listing documents: {e}")


def import_slack(args):
    """Import Slack conversations and messages."""
    try:
        importer = SlackImporter()
        docs = importer.import_slack(args.filter, args.limit)
        if docs:
            config = ConfigLoader.load("config/vatuta.yaml")
            doc_manager = QdrantDocumentManager(config.qdrant)
            success = doc_manager.add_documents(docs)
            if success:
                print(f"✅ Successfully imported {len(docs)} Slack messages")
            else:
                print("❌ Failed to save Slack documents")
        else:
            print("⚠️ No Slack messages found")
    except Exception as e:
        print(f"❌ Error importing Slack: {e}")


def import_gitlab(args):
    """Import GitLab issues and/or merge requests."""
    try:
        importer = GitLabImporter()
        project_ids = [int(p) for p in args.project_ids.split(",")] if args.project_ids else None
        docs = importer.import_gitlab(
            project_ids=project_ids,
            include=args.include,
            state=args.state,
            limit_per_project=args.limit,
        )
        if docs:
            config = ConfigLoader.load("config/vatuta.yaml")
            doc_manager = QdrantDocumentManager(config.qdrant)
            success = doc_manager.add_documents(docs)
            if success:
                print(f"✅ Successfully imported {len(docs)} GitLab documents")
            else:
                print("❌ Failed to save GitLab documents")
        else:
            print("⚠️ No GitLab documents found")
    except Exception as e:
        print(f"❌ Error importing GitLab: {e}")


def show_stats(args):
    """Show knowledge base statistics."""
    try:
        config = ConfigLoader.load("config/vatuta.yaml")
        doc_manager = QdrantDocumentManager(config.qdrant)
        stats = doc_manager.get_document_stats()

        print("📊 Knowledge Base Statistics:")
        print(f"  Total documents: {stats['total_documents']}")
        print(f"  Storage directory: {stats['storage_directory']}")
        print(f"  Vector store exists: {stats['vectorstore_exists']}")

        if stats["sources"]:
            print("  Sources:")
            for source, count in stats["sources"].items():
                print(f"    - {source}: {count} documents")

    except Exception as e:
        print(f"❌ Error getting stats: {e}")


def clear_documents(args):
    """Clear all documents from the knowledge base."""
    try:
        config = ConfigLoader.load("config/vatuta.yaml")
        doc_manager = QdrantDocumentManager(config.qdrant)
        success = doc_manager.clear_all_documents()

        if success:
            print("✅ All documents cleared from knowledge base")
        else:
            print("❌ Failed to clear documents")

    except Exception as e:
        print(f"❌ Error clearing documents: {e}")


def run_rag(args):
    """Run the simple RAG PoC over the local vector store."""
    # Build argv for rag_poc.main
    argv = [args.question]
    if args.k is not None:
        argv.extend(["--k", str(args.k)])
    if args.model_id:
        argv.extend(["--model-id", args.model_id])
    if args.profile:
        argv.extend(["--profile", args.profile])
    if args.region:
        argv.extend(["--region", args.region])
    if args.show_sources:
        argv.append("--show-sources")
    if getattr(args, "show_kb_stats", False):
        argv.append("--show-kb-stats")
    if getattr(args, "agent", False):
        argv.append("--agent")

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]] + argv
        rag_main()
    finally:
        sys.argv = old_argv


def run_tools_demo(args):
    """Run the simple tools demo (kb_stats and search_kb)."""

    argv = [args.question]
    if args.k is not None:
        argv.extend(["--k", str(args.k)])
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]] + argv
        tools_main()
    finally:
        sys.argv = old_argv


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Document Management CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Import JIRA tickets command
    import_jira_parser = subparsers.add_parser("import-jira", help="Import JIRA tickets")
    import_jira_parser.add_argument(
        "--jql",
        default="project = PROJ AND issuetype = Epic AND statusCategory != Done",
        help="JQL query for ticket search",
    )
    import_jira_parser.add_argument("--max-results", type=int, default=100, help="Maximum number of results")
    import_jira_parser.set_defaults(func=import_jira_tickets)

    # Import Confluence pages command
    import_confluence_parser = subparsers.add_parser("import-confluence", help="Import Confluence pages")
    import_confluence_parser.add_argument("--space-key", default="RDT", help="Confluence space key")
    import_confluence_parser.add_argument("--limit", type=int, default=100, help="Maximum number of pages")
    import_confluence_parser.set_defaults(func=import_confluence_pages)

    # Search documents command
    search_parser = subparsers.add_parser("search", help="Search documents")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--k", type=int, default=5, help="Number of results to return")
    search_parser.set_defaults(func=search_documents)

    # List documents command
    list_parser = subparsers.add_parser("list", help="List documents")
    list_parser.add_argument("--source", help="Filter by source (jira, confluence)")
    list_parser.add_argument("--limit", type=int, default=10, help="Maximum number of documents to show")
    list_parser.set_defaults(func=list_documents)

    # Import Slack command
    slack_parser = subparsers.add_parser("import-slack", help="Import Slack conversations")
    slack_parser.add_argument("--filter", default=None, help="Substring to filter channel names")
    slack_parser.add_argument("--limit", type=int, default=500, help="Max messages per channel")
    slack_parser.set_defaults(func=import_slack)

    # Import GitLab command
    gitlab_parser = subparsers.add_parser("import-gitlab", help="Import GitLab issues and MRs")
    gitlab_parser.add_argument(
        "--project-ids",
        default=None,
        help="Comma-separated project IDs; defaults to GITLAB_PROJECT_IDS",
    )
    gitlab_parser.add_argument(
        "--include",
        default="both",
        choices=["issues", "mrs", "merge_requests", "both"],
        help="What to import",
    )
    gitlab_parser.add_argument("--state", default=None, help="State filter (e.g., opened, closed, merged)")
    gitlab_parser.add_argument("--limit", type=int, default=100, help="Max items per project per type")
    gitlab_parser.set_defaults(func=import_gitlab)

    # Show stats command
    stats_parser = subparsers.add_parser("stats", help="Show knowledge base statistics")
    stats_parser.set_defaults(func=show_stats)

    # Clear documents command
    clear_parser = subparsers.add_parser("clear", help="Clear all documents")
    clear_parser.set_defaults(func=clear_documents)

    # RAG PoC query command
    rag_parser = subparsers.add_parser("rag", help="Query the RAG PoC (uses Bedrock)")
    rag_parser.add_argument("question", help="User question to answer")
    rag_parser.add_argument("--k", type=int, default=4, help="Top-K documents to retrieve")
    rag_parser.add_argument(
        "--model-id",
        default="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        help="Bedrock model id",
    )
    rag_parser.add_argument("--profile", default=None, help="AWS profile for Bedrock")
    rag_parser.add_argument("--region", default=None, help="AWS region for Bedrock")
    rag_parser.add_argument("--show-sources", action="store_true", help="Show retrieved sources")
    rag_parser.add_argument("--show-kb-stats", action="store_true", help="Show KB stats before answering")
    rag_parser.add_argument("--agent", action="store_true", help="Enable agentic tool-calling (kb_stats)")
    rag_parser.set_defaults(func=run_rag)

    # Tools demo command
    tools_parser = subparsers.add_parser("tools", help="Run tools example (kb_stats, search_kb)")
    tools_parser.add_argument("question", help="Question to try with search_kb")
    tools_parser.add_argument("--k", type=int, default=4, help="Top-K documents to retrieve")
    tools_parser.set_defaults(func=run_tools_demo)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
