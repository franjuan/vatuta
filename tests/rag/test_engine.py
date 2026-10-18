"""Unit tests for RAG retrieval, graph execution, and LLM initialization."""

import unittest
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.models.config import LLMBackendConfig, RagConfig
from src.rag.engine import build_dspy_lm, build_graph, ensure_retriever
from src.rag.qdrant_manager import QdrantDocumentManager


def test_retriever_preserves_search_limit() -> None:
    """Use the existing vector store with the requested result limit."""
    manager = MagicMock(spec=QdrantDocumentManager)
    manager.vectorstore = MagicMock()

    retriever = ensure_retriever(manager, k=7)

    manager.vectorstore.as_retriever.assert_called_once_with(search_kwargs={"k": 7})
    assert retriever is manager.vectorstore.as_retriever.return_value


def test_retriever_requires_vector_store() -> None:
    """Fail clearly when retrieval has no vector store."""
    manager = MagicMock(spec=QdrantDocumentManager)
    manager.vectorstore = None

    with pytest.raises(ValueError, match="No vector store found"):
        ensure_retriever(manager, k=4)


def test_graph_passes_retrieved_context_to_generation() -> None:
    """Retrieve documents and generate an answer without external services."""
    manager = MagicMock(spec=QdrantDocumentManager)
    manager.vectorstore = MagicMock()
    retriever = manager.vectorstore.as_retriever.return_value
    documents = [Document(page_content="First fact"), Document(page_content="Second fact")]
    retriever.invoke.return_value = documents
    generator = MagicMock()
    generator.return_value.answer = "Grounded answer"

    result = build_graph(manager, k=2, rag_module=generator).invoke({"question": "What is known?"})

    retriever.invoke.assert_called_once_with("What is known?")
    generator.assert_called_once_with(question="What is known?", context="First fact\n\nSecond fact")
    assert result["docs"] == documents
    assert result["answer"] == "Grounded answer"


class TestEngineLM(unittest.TestCase):
    """Test suite for RAG engine build_dspy_lm delegation."""

    def setUp(self) -> None:
        """Create mock RagConfig for testing."""
        self.config = RagConfig(
            llm_backends={
                "gemini_flash": LLMBackendConfig(
                    model="gemini/gemini-2.5-flash",
                    temperature=0.2,
                    max_tokens=800,
                ),
            },
            router_backend="gemini_flash",
            generator_backend="gemini_flash",
        )

    @patch("src.rag.engine.LLMProviderManager")
    def test_build_dspy_lm_delegates_to_provider_manager(self, mock_manager_cls: MagicMock) -> None:
        """Verify build_dspy_lm delegates instantiation to LLMProviderManager."""
        mock_instance = MagicMock()
        mock_lm = MagicMock()
        mock_instance.get_dspy_lm.return_value = mock_lm
        mock_manager_cls.return_value = mock_instance

        result = build_dspy_lm(self.config, "gemini_flash")

        mock_manager_cls.assert_called_once_with(self.config)
        mock_instance.get_dspy_lm.assert_called_once_with("gemini_flash", role="general")
        self.assertEqual(result, mock_lm)

    @patch("src.rag.engine.LLMProviderManager")
    def test_build_dspy_lm_with_role(self, mock_manager_cls: MagicMock) -> None:
        """Verify build_dspy_lm passes role parameter if provided."""
        mock_instance = MagicMock()
        mock_lm = MagicMock()
        mock_instance.get_dspy_lm.return_value = mock_lm
        mock_manager_cls.return_value = mock_instance

        result = build_dspy_lm(self.config, "gemini_flash", role="router")

        mock_instance.get_dspy_lm.assert_called_once_with("gemini_flash", role="router")
        self.assertEqual(result, mock_lm)

    @patch("src.rag.engine.LLMProviderManager")
    def test_distinct_models_per_role(self, mock_manager_cls: MagicMock) -> None:
        """Verify router and generator backends can resolve to different models and roles."""
        multi_config = RagConfig(
            llm_backends={
                "gemini_flash": LLMBackendConfig(model="gemini/gemini-2.5-flash"),
                "gemini_pro": LLMBackendConfig(model="gemini/gemini-1.5-pro"),
            },
            router_backend="gemini_flash",
            generator_backend="gemini_pro",
        )
        mock_instance = MagicMock()
        mock_router_lm = MagicMock()
        mock_generator_lm = MagicMock()
        mock_instance.get_dspy_lm.side_effect = lambda backend, role: (
            mock_router_lm if role == "router" else mock_generator_lm
        )
        mock_manager_cls.return_value = mock_instance

        router_lm = build_dspy_lm(multi_config, multi_config.router_backend, role="router")
        generator_lm = build_dspy_lm(multi_config, multi_config.generator_backend, role="generator")

        self.assertEqual(router_lm, mock_router_lm)
        self.assertEqual(generator_lm, mock_generator_lm)
        self.assertNotEqual(router_lm, generator_lm)
