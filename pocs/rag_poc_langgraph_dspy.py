"""
RAG PoC using LangGraph for context flow and DSPy for prompts (Bedrock models)

This module reuses the FAISS vector store from `DocumentManager` for retrieval,
executes a small LangGraph over the RAG state (retrieve -> generate), and uses
DSPy to prompt an Anthropic model on Bedrock for the final answer.

Usage (module):
  python -m pocs.rag_poc_langgraph_dspy "What are the epic requirements?" --k 4 --show-sources

Environment:
  - ANTHROPIC_API_KEY must be set (used by Bedrock/Anthropic via DSPy LM)
  - AWS_PROFILE (default: IAAdmin) and AWS_REGION (default: us-east-1)
    are used directly by the SDK resolution
"""

from __future__ import annotations

import argparse
import os
from typing import List, TypedDict

import dspy
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph

from pocs.tools_example import kb_stats
from src.rag.document_manager import DocumentManager


class RAGState(TypedDict, total=False):
    question: str
    k: int
    docs: List[Document]
    context: str
    answer: str


def ensure_retriever(document_manager: DocumentManager, k: int):
    if document_manager.vectorstore is None:
        raise ValueError("No vector store found. Import documents first (see doc_cli commands).")
    return document_manager.vectorstore.as_retriever(search_kwargs={"k": k})


def format_docs(documents: List[Document]) -> str:
    return "\n\n".join(d.page_content for d in documents)


class RAGAnswer(dspy.Signature):
    """Answer concisely using only the provided context; admit if unknown."""

    context: str = dspy.InputField(desc="Relevant context from retrieved documents")
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="Concise answer grounded in the context")


class DSPyRAGModule(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.generate = dspy.ChainOfThought(RAGAnswer)

    def forward(self, question: str, context: str) -> dspy.Prediction:
        pred = self.generate(context=context, question=question)
        return dspy.Prediction(answer=pred.answer)


def configure_dspy_lm(model_id: str, temperature: float = 0.2, max_tokens: int = 800) -> None:
    # Region is required
    region = os.getenv("AWS_REGION", "us-east-1")
    os.environ["AWS_REGION"] = region

    if os.getenv("AWS_BEARER_TOKEN_BEDROCK"):
        os.environ.pop("AWS_PROFILE", None)
    else:
        profile = os.getenv("AWS_PROFILE")
        if not profile:
            raise RuntimeError("Set AWS_BEARER_TOKEN_BEDROCK (bearer) or AWS_PROFILE (profile).")
        os.environ["AWS_PROFILE"] = profile

    lm = dspy.LM(model=model_id, temperature=temperature, max_tokens=max_tokens)
    dspy.configure(lm=lm)


def build_graph(document_manager: DocumentManager, k: int, rag_module: DSPyRAGModule):
    graph = StateGraph(RAGState)

    def retrieve_node(state: RAGState) -> RAGState:
        retriever = ensure_retriever(document_manager, k=k)
        docs: List[Document] = retriever.invoke(state["question"])  # type: ignore[index]
        return {"docs": docs, "context": format_docs(docs)}

    def generate_node(state: RAGState) -> RAGState:
        context_text = state.get("context", "")
        pred = rag_module(question=state["question"], context=context_text)  # type: ignore[index]
        return {"answer": getattr(pred, "answer", str(pred))}

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG with LangGraph (context) + DSPy (prompts)")
    parser.add_argument("question", help="User question to answer with RAG")
    parser.add_argument("--k", type=int, default=4, help="Top-K documents to retrieve")
    parser.add_argument(
        "--model-id",
        default="bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        help="Model id for DSPy",
    )
    parser.add_argument(
        "--show-sources",
        action="store_true",
        help="Print brief info about retrieved source documents",
    )
    parser.add_argument(
        "--show-kb-stats",
        action="store_true",
        help="Print knowledge base stats (via kb_stats tool) before answering",
    )
    parser.add_argument(
        "--ingest-slack",
        action="store_true",
        help="Ingest cached Slack data before answering",
    )
    parser.add_argument(
        "--ingest-jira",
        action="store_true",
        help="Ingest cached JIRA data before answering",
    )
    parser.add_argument(
        "--ingest-confluence",
        action="store_true",
        help="Ingest cached Confluence data before answering",
    )
    args = parser.parse_args()

    doc_manager = DocumentManager()

    if args.ingest_slack:
        ingest_slack_cache(doc_manager)

    if args.ingest_jira:
        ingest_jira_cache(doc_manager)

    if args.ingest_confluence:
        ingest_confluence_cache(doc_manager)

    if args.show_kb_stats:
        print("📊 KB Stats:")
        try:
            print(kb_stats.invoke({}))
        except Exception:
            print("(failed to get stats)")

    try:
        # Ensure the store is present early to provide a friendly error
        ensure_retriever(doc_manager, k=args.k)
    except ValueError as e:
        print(f"❌ {e}")
        return

    configure_dspy_lm(model_id=args.model_id)
    rag_module = DSPyRAGModule()
    app = build_graph(doc_manager, args.k, rag_module)

    print("🤖 Question:", args.question)
    final_state: RAGState = app.invoke({"question": args.question, "k": args.k})
    print("\n✅ Answer:\n")
    print(final_state.get("answer", ""))

    if args.show_sources:
        docs = final_state.get("docs", []) or []
        print("\n📚 Top retrieved sources:\n")
        for i, d in enumerate(docs, start=1):
            title = d.metadata.get("title") or d.metadata.get("summary") or "Untitled"
            source = d.metadata.get("source", "unknown")
            preview = d.page_content[:120].replace("\n", " ") + ("..." if len(d.page_content) > 120 else "")
            print(f"  {i}. {title} ({source})")
            print(f"     {preview}")


def ingest_slack_cache(doc_manager: DocumentManager) -> None:
    """Ingest cached Slack data into the document manager."""
    from src.sources.slack import SlackSource

    print("📥 Ingesting Slack cache...")

    # Check for token
    if not os.getenv("SLACK_BOT_TOKEN"):
        print("⚠️ SLACK_BOT_TOKEN not set. SlackSource requires it even for cache reading.")
        print("   Please set SLACK_BOT_TOKEN to proceed with ingestion.")
        return

    try:
        # Initialize SlackSource with paths matching the project structure
        slack_source = SlackSource(
            config={
                "channel_types": ["public_channel", "private_channel", "im", "mpim"],
                "workspace_domain": "https://devoinc.slack.com",  # Defaulting to what was seen in slack.py
                "user_cache_path": os.path.join("data", "slack_users_cache.json"),
                "storage_path": os.path.join("data", "slack"),
            },
            secrets={"bot_token": os.getenv("SLACK_BOT_TOKEN")},
            id="slack-main",
        )

        # Collect cached documents
        docs, chunks = slack_source.collect_cached_documents_and_chunks()

        if not docs:
            print("⚠️ No cached Slack documents found.")
            return

        print(f"   Found {len(docs)} documents and {len(chunks)} chunks.")

        # Create a map of documents by ID for chunk processing
        docs_by_id = {d.document_id: d for d in docs}

        # Add to DocumentManager
        # Note: We pass chunks to add_chunk_records, which handles vector store addition
        doc_manager.add_chunk_records(chunks, docs_by_id)

    except Exception as e:
        print(f"❌ Error ingesting Slack cache: {e}")


def ingest_jira_cache(doc_manager: DocumentManager) -> None:
    """Ingest cached JIRA data into the document manager."""
    from src.sources.jira import JiraSource

    print("📥 Ingesting JIRA cache...")

    # Check for credentials
    jira_user = os.getenv("JIRA_USER")
    jira_token = os.getenv("JIRA_API_TOKEN")

    # We need credentials to initialize the source, even if reading from cache
    if not jira_user or not jira_token:
        print("⚠️ JIRA_USER or JIRA_API_TOKEN not set. JiraSource requires them even for cache reading.")
        print("   Please set them to proceed with ingestion.")
        return

    try:
        # Initialize JiraSource
        # We provide a dummy URL and empty projects list as we rely on cache discovery
        jira_source = JiraSource(
            config={
                "url": os.getenv("JIRA_INSTANCE_URL", "https://placeholder.atlassian.net"),
                "projects": [],  # Empty list, cache collection will discover projects from directories
                "storage_path": os.path.join("data", "jira"),
            },
            secrets={
                "jira_user": jira_user,
                "jira_api_token": jira_token,
            },
            id="jira-main",
        )

        # Collect cached documents
        docs, chunks = jira_source.collect_cached_documents_and_chunks()

        if not docs:
            print("⚠️ No cached JIRA documents found.")
            return

        print(f"   Found {len(docs)} documents and {len(chunks)} chunks.")

        # Create a map of documents by ID for chunk processing
        docs_by_id = {d.document_id: d for d in docs}

        # Add to DocumentManager
        doc_manager.add_chunk_records(chunks, docs_by_id)

    except Exception as e:
        print(f"❌ Error ingesting JIRA cache: {e}")


def ingest_confluence_cache(doc_manager: DocumentManager) -> None:
    """Ingest cached Confluence data into the document manager."""
    from src.sources.confluence import ConfluenceSource

    print("📥 Ingesting Confluence cache...")

    # Check for credentials
    jira_user = os.getenv("JIRA_USER")
    jira_token = os.getenv("JIRA_API_TOKEN")

    # We need credentials to initialize the source, even if reading from cache
    if not jira_user or not jira_token:
        print("⚠️ JIRA_USER or JIRA_API_TOKEN not set. ConfluenceSource requires them even for cache reading.")
        print("   Please set them to proceed with ingestion.")
        return

    try:
        # Initialize ConfluenceSource
        # We provide a dummy URL and empty spaces list as we rely on cache discovery
        confluence_source = ConfluenceSource(
            config={
                "url": os.getenv("JIRA_INSTANCE_URL", "https://placeholder.atlassian.net"),
                "spaces": [],  # Empty list, cache collection will discover spaces from directories
                "storage_path": os.path.join("data", "confluence"),
            },
            secrets={
                "jira_user": jira_user,
                "jira_api_token": jira_token,
            },
            id="confluence-main",
        )

        # Collect cached documents
        docs, chunks = confluence_source.collect_cached_documents_and_chunks()

        if not docs:
            print("⚠️ No cached Confluence documents found.")
            return

        print(f"   Found {len(docs)} documents and {len(chunks)} chunks.")

        # Create a map of documents by ID for chunk processing
        docs_by_id = {d.document_id: d for d in docs}

        # Add to DocumentManager
        doc_manager.add_chunk_records(chunks, docs_by_id)

    except Exception as e:
        print(f"❌ Error ingesting Confluence cache: {e}")


if __name__ == "__main__":
    main()
