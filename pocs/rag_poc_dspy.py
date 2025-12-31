"""
Simple RAG PoC using DSPy and Anthropic

This module loads the existing FAISS vector store persisted by `DocumentManager`
and exposes a minimal Retrieval-Augmented Generation (RAG) flow built with DSPy.

Usage (module):
    python -m pocs.rag_poc_dspy "What are the epic requirements?" --k 4 --show-sources

Environment:
    - ANTHROPIC_API_KEY must be set to use Anthropic models via DSPy

Notes:
    - Retrieval reuses the existing `DocumentManager` FAISS store.
    - Generation is performed via DSPy using an Anthropic LM.
"""

from __future__ import annotations

import argparse
import os
from typing import List, Union

import dspy
from langchain_core.documents import Document

from pocs.tools_example import kb_stats
from src.models.config import ConfigLoader
from src.rag.document_manager import DocumentManager
from src.rag.qdrant_manager import QdrantDocumentManager


def ensure_retriever(document_manager: Union[DocumentManager, QdrantDocumentManager], k: int):
    """Return a retriever from the existing FAISS vector store.

    Raises ValueError if the store is not available.
    """
    if document_manager.vectorstore is None:
        raise ValueError("No vector store found. Import documents first (see doc_cli commands).")
    return document_manager.vectorstore.as_retriever(search_kwargs={"k": k})


def format_docs(documents: List[Document]) -> str:
    """Format retrieved documents as context text."""
    return "\n\n".join(d.page_content for d in documents)


class RAGAnswer(dspy.Signature):
    """Answer with the provided context; say you don't know if missing."""

    context: str = dspy.InputField(desc="Relevant context from retrieved documents")
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="Concise answer grounded in the context")


class SimpleRAG(dspy.Module):
    """Minimal DSPy RAG module that generates an answer from context and question."""

    def __init__(self) -> None:
        super().__init__()
        # ChainOfThought can yield better grounding; switch to Predict for shorter outputs
        self.generate = dspy.ChainOfThought(RAGAnswer)

    # Disallow access-key auth in this script
    if os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("AWS_SECRET_ACCESS_KEY"):
        raise RuntimeError("Access keys are not supported here. Use AWS_BEARER_TOKEN_BEDROCK or an AWS profile.")

    def forward(self, question: str, context: str) -> dspy.Prediction:
        prediction = self.generate(context=context, question=question)
        return dspy.Prediction(answer=prediction.answer)


def configure_dspy_lm(
    model_id: str = "bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    temperature: float = 0.2,
    max_tokens: int = 800,
) -> None:
    """Configure DSPy with Bedrock using ONLY two auth methods: bearer token or profile.

    - Bearer token: set `AWS_BEARER_TOKEN_BEDROCK` (see AWS docs)
      https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys-use.html
    - Profile: set `AWS_PROFILE`

    Access keys are not supported in this script.
    """

    # Region is required
    region = os.getenv("AWS_REGION", "us-east-1")
    os.environ["AWS_REGION"] = region

    if os.getenv("AWS_BEARER_TOKEN_BEDROCK"):
        # Use bearer token; clear profile to avoid ambiguity
        os.environ.pop("AWS_PROFILE", None)
    else:
        profile = os.getenv("AWS_PROFILE")
        if not profile:
            raise RuntimeError("Set AWS_BEARER_TOKEN_BEDROCK (bearer) or AWS_PROFILE (profile).")
        os.environ["AWS_PROFILE"] = profile

    lm = dspy.LM(model=model_id, temperature=temperature, max_tokens=max_tokens)
    dspy.configure(lm=lm)


def build_kb_stats_tool() -> dspy.Tool:
    """Wrap the existing kb_stats into a DSPy Tool."""

    def _kb_stats_py() -> str:
        """Return basic knowledge base stats: total docs and sources."""
        try:
            return kb_stats.invoke({})
        except Exception:
            return "(tool kb_stats failed)"

    return dspy.Tool(
        _kb_stats_py,
        name="kb_stats",
        desc="Get knowledge base stats (total docs and source counts).",
    )


class QAReActWithContext(dspy.Signature):
    """Answer using retrieved context; may call tools if needed."""

    context: str = dspy.InputField(desc="Relevant context from retrieved documents")
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="Concise, factual answer grounded in context")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple DSPy RAG PoC (Anthropic)")
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
        "--agent",
        action="store_true",
        help="Enable ReAct agent with kb_stats tool (model may call it)",
    )
    parser.add_argument(
        "--react-iters",
        type=int,
        default=8,
        help="Max ReAct reasoning/tool iterations",
    )
    args = parser.parse_args()

    config = ConfigLoader.load("config/vatuta.yaml")
    doc_manager = QdrantDocumentManager(config.qdrant)

    if args.show_kb_stats:
        print("📊 KB Stats:")
        try:
            print(kb_stats.invoke({}))
        except Exception:
            print("(failed to get stats)")

    try:
        retriever = ensure_retriever(doc_manager, k=args.k)
    except ValueError as e:
        print(f"❌ {e}")
        return

    # Configure DSPy LM and build module
    configure_dspy_lm(model_id=args.model_id)
    print("🤖 Question:", args.question)

    # Retrieve context
    docs: List[Document] = retriever.invoke(args.question)
    context_text = format_docs(docs)

    if args.agent:
        # ReAct path with kb_stats tool available
        tool = build_kb_stats_tool()
        react = dspy.ReAct(QAReActWithContext, tools=[tool], max_iters=args.react_iters)
        result = react(question=args.question, context=context_text)
        answer_text = getattr(result, "answer", str(result))
    else:
        # Simple non-agentic DSPy generation
        rag = SimpleRAG()
        result = rag(question=args.question, context=context_text)
        answer_text = getattr(result, "answer", str(result))

    print("\n✅ Answer:\n")
    print(answer_text)

    if args.show_sources:
        print("\n📚 Top retrieved sources:\n")
        for i, d in enumerate(docs, start=1):
            title = d.metadata.get("title") or d.metadata.get("summary") or "Untitled"
            source = d.metadata.get("source", "unknown")
            preview = d.page_content[:120].replace("\n", " ") + ("..." if len(d.page_content) > 120 else "")
            print(f"  {i}. {title} ({source})")
            print(f"     {preview}")


if __name__ == "__main__":
    main()
