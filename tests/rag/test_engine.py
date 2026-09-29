"""Unit tests for RAG engine LLM initialization delegating to LLMProviderManager."""

import unittest
from unittest.mock import MagicMock, patch

from src.models.config import LLMBackendConfig, RagConfig
from src.rag.engine import build_dspy_lm


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
