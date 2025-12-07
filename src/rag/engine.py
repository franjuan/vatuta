"""RAG engine using LangGraph and DSPy.

Implements the retrieval-augmented generation system.
"""

from __future__ import annotations

import os
from typing import Any, List, TypedDict

import dspy
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph

from src.models.config import RagConfig
from src.rag.document_manager import DocumentManager


class RAGState(TypedDict, total=False):
    """State dictionary for RAG graph execution."""

    question: str
    k: int
    docs: List[Document]
    context: str
    answer: str


def ensure_retriever(document_manager: DocumentManager, k: int) -> Any:
    """Ensure vector store exists and return retriever."""
    if document_manager.vectorstore is None:
        raise ValueError("No vector store found. Import documents first (see doc_cli commands).")
    return document_manager.vectorstore.as_retriever(search_kwargs={"k": k})


def format_docs(documents: List[Document]) -> str:
    """Format list of documents into a single string."""
    return "\n\n".join(d.page_content for d in documents)


class RAGAnswer(dspy.Signature):
    """Answer concisely using only the provided context; admit if unknown."""

    context: str = dspy.InputField(desc="Relevant context from retrieved documents")
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="Concise answer grounded in the context")


class DSPyRAGModule(dspy.Module):
    """DSPy module for RAG answer generation."""

    def __init__(self) -> None:
        """Initialize the RAG module."""
        super().__init__()
        self.generate = dspy.ChainOfThought(RAGAnswer)

    def forward(self, question: str, context: str) -> dspy.Prediction:
        """Generate answer from question and context."""
        pred = self.generate(context=context, question=question)
        return dspy.Prediction(answer=getattr(pred, "answer", ""))


def configure_dspy_lm(config: RagConfig) -> None:
    """Configure DSPy language model with AWS Bedrock."""
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

    lm = dspy.LM(model=config.model_id, temperature=config.temperature, max_tokens=config.max_tokens)
    dspy.configure(lm=lm)


def build_graph(document_manager: DocumentManager, k: int, rag_module: DSPyRAGModule) -> Any:
    """Build LangGraph workflow for RAG execution."""
    graph = StateGraph(RAGState)

    def retrieve_node(state: RAGState) -> RAGState:
        retriever = ensure_retriever(document_manager, k=k)
        docs: List[Document] = retriever.invoke(state["question"])
        return {"docs": docs, "context": format_docs(docs)}

    def generate_node(state: RAGState) -> RAGState:
        context_text = state.get("context", "")
        pred = rag_module(question=state["question"], context=context_text)
        return {"answer": getattr(pred, "answer", str(pred))}

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()
