"""
Tools Example (LangChain 1.0)

This module defines two example tools and a small agent-like loop to call them:
- search_kb: uses the existing FAISS retriever to search the knowledge base
- kb_stats: returns basic stats from the DocumentManager

Run (module):
    python -m pocs.tools_example "What are epic requirements?" --k 4

Note: This example does not implement full agent planning; it demonstrates how
LangChain tools are modeled and invoked programmatically.
"""

import argparse
from typing import List

from langchain_core.documents import Document
from langchain_core.tools import tool

from src.rag.document_manager import DocumentManager


def _ensure_doc_manager() -> DocumentManager:
    return DocumentManager()


def _ensure_retriever(k: int):
    dm = _ensure_doc_manager()
    if dm.vectorstore is None:
        raise ValueError("No vector store found. Import documents first.")
    return dm.vectorstore.as_retriever(search_kwargs={"k": k})


def _format_docs(docs: List[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)


@tool("search_kb", return_direct=False)
def search_kb(query: str, k: int = 4) -> str:
    """Search the local knowledge base using semantic similarity. Args: query, k."""
    retriever = _ensure_retriever(k)
    docs = retriever.invoke(query)
    return _format_docs(docs)


@tool("kb_stats", return_direct=True)
def kb_stats() -> str:
    """Return basic knowledge base stats: total docs and sources."""
    dm = _ensure_doc_manager()
    stats = dm.get_document_stats()
    sources = ", ".join(f"{s}:{c}" for s, c in stats.get("sources", {}).items())
    return f"total={stats.get('total_documents', 0)}, sources=[{sources}]"


def main():
    parser = argparse.ArgumentParser(description="LangChain 1.0 Tools Example")
    parser.add_argument("question", help="Question to try with search_kb tool")
    parser.add_argument("--k", type=int, default=4, help="Top-K for retrieval")
    args = parser.parse_args()

    # Simple demonstration: call kb_stats, then search_kb
    print("🔧 Tool: kb_stats")
    print(kb_stats.invoke({}))

    print("\n🔧 Tool: search_kb -> formatted context")
    context = search_kb.invoke({"query": args.question, "k": args.k})
    print(context[:800] + ("..." if len(context) > 800 else ""))


if __name__ == "__main__":
    main()
