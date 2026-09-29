"""Unit tests for LLMProviderManager factory and lifecycle management."""

import unittest
from unittest.mock import MagicMock, patch

from src.llm.errors import LLMConfigurationError
from src.llm.provider import LLMProviderManager
from src.models.config import LLMBackendConfig, RagConfig


class TestLLMProviderManager(unittest.TestCase):
    """Test suite for LLMProviderManager."""

    def setUp(self) -> None:
        """Create mock RagConfig for testing."""
        self.config = RagConfig(
            llm_backends={
                "gemini_flash": LLMBackendConfig(
                    model="gemini/gemini-2.5-flash",
                    temperature=0.2,
                    max_tokens=800,
                ),
                "gemini_pro": LLMBackendConfig(
                    model="gemini/gemini-1.5-pro",
                    temperature=0.7,
                    max_tokens=2000,
                ),
            },
            router_backend="gemini_flash",
            generator_backend="gemini_pro",
        )
        self.manager = LLMProviderManager(self.config)

    @patch("litellm.validate_environment")
    def test_validate_backends_success(self, mock_validate: MagicMock) -> None:
        """Verify successful validation when all backends have valid credentials."""
        mock_validate.return_value = {"keys_in_environment": True, "missing_keys": []}
        # Should not raise
        self.manager.validate_backends()
        self.assertEqual(mock_validate.call_count, 2)

    @patch("litellm.validate_environment")
    def test_validate_backends_missing_credentials(self, mock_validate: MagicMock) -> None:
        """Verify validation raises LLMConfigurationError on missing environment keys."""
        mock_validate.return_value = {
            "keys_in_environment": False,
            "missing_keys": ["GEMINI_API_KEY"],
        }
        with self.assertRaises(LLMConfigurationError) as ctx:
            self.manager.validate_backends()
        self.assertIn("GEMINI_API_KEY", str(ctx.exception))

    @patch("dspy.LM")
    def test_get_dspy_lm_caching(self, mock_dspy_lm: MagicMock) -> None:
        """Verify get_dspy_lm instantiates and caches dspy.LM instances per backend and role."""
        mock_instance = MagicMock()
        mock_dspy_lm.return_value = mock_instance

        lm1 = self.manager.get_dspy_lm("gemini_flash", role="router")
        lm2 = self.manager.get_dspy_lm("gemini_flash", role="router")

        self.assertIs(lm1, lm2)
        mock_dspy_lm.assert_called_once()

    @patch("dspy.LM")
    def test_get_dspy_lm_distinct_roles(self, mock_dspy_lm: MagicMock) -> None:
        """Verify distinct roles produce distinct cached dspy.LM instances for telemetry."""
        mock_dspy_lm.side_effect = lambda *args, **kwargs: MagicMock()

        lm_router = self.manager.get_dspy_lm("gemini_flash", role="router")
        lm_gen = self.manager.get_dspy_lm("gemini_flash", role="generator")

        self.assertIsNot(lm_router, lm_gen)
        self.assertEqual(mock_dspy_lm.call_count, 2)

    def test_get_dspy_lm_unknown_backend(self) -> None:
        """Verify requesting unconfigured backend raises LLMConfigurationError."""
        with self.assertRaises(LLMConfigurationError):
            self.manager.get_dspy_lm("non_existent_backend")
