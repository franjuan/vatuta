"""Module for the RAG engine implementation."""

from __future__ import annotations

import os
from typing import Any, List, Optional, TypedDict, Union

import dspy
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph

from src.models.config import RagConfig
from src.rag.document_manager import DocumentManager
from src.rag.qdrant_manager import QdrantDocumentManager


class RAGState(TypedDict, total=False):
    """State dictionary for RAG graph execution."""

    question: str
    k: int
    docs: List[Document]
    context: str
    answer: str


def ensure_retriever(document_manager: Union[DocumentManager, QdrantDocumentManager], k: int) -> Any:
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


def configure_dspy_lm(config: RagConfig, backend_name: Optional[str] = None) -> None:
    """Configure DSPy language model based on backend selection.

    Args:
        config: RAG configuration object.
        backend_name: Name of the backend to use. If None, uses the first configured backend.
    """
    if not config.llm_backend:
        raise ValueError("No LLM backends configured in config.rag.llm_backend")

    # Select backend
    if backend_name:
        if backend_name not in config.llm_backend:
            # Should we error or default? Failing is safer if explicit request made.
            # Check if user meant a type or an ID.
            # For now, strict match.
            available = ", ".join(config.llm_backend.keys())
            raise ValueError(f"Backend '{backend_name}' not found. Available: {available}")
        selection = backend_name
    else:
        # Default to first
        selection = next(iter(config.llm_backend.keys()))

    print(f"Using LLM backend: {selection}")  # Simple print for CLI visibility, client.py will also log

    llm_conf = config.llm_backend[selection]

    # Configure based on selection or model_id characteristics
    # We can infer provider from backend name (e.g. 'gemini', 'bedrock') or model_id

    is_gemini = "gemini" == selection.lower()
    is_bedrock = "bedrock" == selection.lower()

    if is_gemini:
        # Gemini Configuration
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable not set.")

        lm = dspy.LM(
            llm_conf.model_id,
            api_key=api_key,
            max_tokens=llm_conf.max_tokens,
            temperature=llm_conf.temperature,
        )
        dspy.configure(lm=lm)

    elif is_bedrock:
        # Default / Bedrock Configuration
        # Region is required for Bedrock
        region = os.getenv("AWS_REGION", "us-east-1")
        os.environ["AWS_REGION"] = region

        if os.getenv("AWS_BEARER_TOKEN_BEDROCK"):
            os.environ.pop("AWS_PROFILE", None)
        else:
            profile = os.getenv("AWS_PROFILE")
            if not profile:
                # Making this optional if using default creds chain? Original code enforced it.
                # Keeping original enforcement for safety.
                raise RuntimeError("Set AWS_BEARER_TOKEN_BEDROCK (bearer) or AWS_PROFILE (profile).")
            os.environ["AWS_PROFILE"] = profile

        lm = dspy.LM(model=llm_conf.model_id, temperature=llm_conf.temperature, max_tokens=llm_conf.max_tokens)
        dspy.configure(lm=lm)

    else:
        raise ValueError(
            f"Backend '{selection}' with model '{llm_conf.model_id}' is not supported. Must be 'bedrock' or 'gemini'."
        )


def build_graph(
    document_manager: Union[DocumentManager, QdrantDocumentManager], k: int, rag_module: DSPyRAGModule
) -> Any:
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
