"""Unit tests for CLI client pre-flight checks and error handling."""

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from src.client.client import app
from src.llm.errors import (
    LLMAuthenticationError,
    LLMRateLimitError,
)
from src.models.config import LLMBackendConfig, RagConfig, VatutaConfig


def _create_mock_config(model: str = "google/gemini-2.5-flash") -> VatutaConfig:
    """Create a minimal VatutaConfig instance for CLI tests.

    Args:
        model: Model identifier string.

    Returns:
        Configured VatutaConfig mock instance.
    """
    mock_cfg = MagicMock(spec=VatutaConfig)
    rag_conf = MagicMock(spec=RagConfig)
    backend_conf = MagicMock(spec=LLMBackendConfig)
    backend_conf.model = model
    backend_conf.temperature = 0.7
    backend_conf.max_tokens = 1000
    backend_conf.extra_kwargs = {}

    rag_conf.llm_backends = {"gemini": backend_conf}
    rag_conf.router_backend = "gemini"
    rag_conf.generator_backend = "gemini"

    mock_cfg.rag = rag_conf
    mock_cfg.entities_manager = MagicMock()
    mock_cfg.entities_manager.storage_path = "data/entities.json"
    mock_cfg.qdrant = MagicMock()
    mock_cfg.qdrant.embeddings.model = "sentence-transformers/all-MiniLM-L6-v2"
    return mock_cfg


def test_cli_preflight_missing_credentials_fails_cleanly() -> None:
    """Test startup pre-flight check terminates with exit code 1 on missing credentials."""
    runner = CliRunner()
    mock_cfg = _create_mock_config()

    with patch("src.client.client.ConfigLoader.load", return_value=mock_cfg):
        with patch("src.client.client.QdrantDocumentManager"):
            with patch(
                "src.llm.provider.litellm.validate_environment",
                return_value={
                    "keys_in_environment": False,
                    "missing_keys": ["GEMINI_API_KEY"],
                },
            ):
                result = runner.invoke(app, ["ask", "What is Vatuta?"])

                assert result.exit_code == 1
                assert "Missing required environment credentials" in result.output
                assert "Traceback (most recent call last)" not in result.output


def test_cli_preflight_malformed_model_fails_cleanly() -> None:
    """Test startup pre-flight check terminates with exit code 1 on malformed model ID."""
    runner = CliRunner()
    mock_cfg = _create_mock_config(model="invalid-model-without-slash")

    with patch("src.client.client.ConfigLoader.load", return_value=mock_cfg):
        with patch("src.client.client.QdrantDocumentManager"):
            result = runner.invoke(app, ["ask", "What is Vatuta?"])

            assert result.exit_code == 1
            assert "provider/model" in result.output
            assert "Traceback (most recent call last)" not in result.output


def test_cli_runtime_ratelimit_error_fails_cleanly() -> None:
    """Test runtime LLMRateLimitError terminates with exit code 1 and clean output."""
    runner = CliRunner()
    mock_cfg = _create_mock_config()

    with patch("src.client.client.ConfigLoader.load", return_value=mock_cfg):
        with patch("src.llm.provider.LLMProviderManager.validate_backends"):
            with patch("src.client.client.QdrantDocumentManager"):
                with patch("src.client.client._get_enabled_sources", return_value=[]):
                    with patch("src.client.client.RAGAgent") as mock_agent_cls:
                        mock_instance = MagicMock()
                        mock_instance.run.side_effect = LLMRateLimitError("Rate limit exceeded. Quota exhausted.")
                        mock_agent_cls.return_value = mock_instance

                        result = runner.invoke(app, ["ask", "What is Vatuta?"])

                        assert result.exit_code == 1
                        assert "Rate limit exceeded" in result.output
                        assert "LLM Runtime Error" in result.output
                        assert "Traceback (most recent call last)" not in result.output


def test_cli_runtime_auth_error_fails_cleanly() -> None:
    """Test runtime LLMAuthenticationError terminates with exit code 1 and clean output."""
    runner = CliRunner()
    mock_cfg = _create_mock_config()

    with patch("src.client.client.ConfigLoader.load", return_value=mock_cfg):
        with patch("src.llm.provider.LLMProviderManager.validate_backends"):
            with patch("src.client.client.QdrantDocumentManager"):
                with patch("src.client.client._get_enabled_sources", return_value=[]):
                    with patch("src.client.client.RAGAgent") as mock_agent_cls:
                        mock_instance = MagicMock()
                        mock_instance.run.side_effect = LLMAuthenticationError(
                            "Authentication failed: Invalid API Key."
                        )
                        mock_agent_cls.return_value = mock_instance

                        result = runner.invoke(app, ["ask", "What is Vatuta?"])

                        assert result.exit_code == 1
                        assert "Authentication failed" in result.output
                        assert "LLM Runtime Error" in result.output
                        assert "Traceback (most recent call last)" not in result.output


def test_cli_ask_success() -> None:
    """Test successful CLI ask execution terminates with exit code 0."""
    runner = CliRunner()
    mock_cfg = _create_mock_config()

    with patch("src.client.client.ConfigLoader.load", return_value=mock_cfg):
        with patch("src.llm.provider.LLMProviderManager.validate_backends"):
            with patch("src.client.client.QdrantDocumentManager"):
                with patch("src.client.client._get_enabled_sources", return_value=[]):
                    with patch("src.client.client.RAGAgent") as mock_agent_cls:
                        mock_instance = MagicMock()
                        mock_instance.run.return_value = {
                            "answer": "Vatuta is a Personal AI Assistant.",
                            "router_cot": {},
                            "generator_cot": "",
                            "routing_summary": "",
                        }
                        mock_agent_cls.return_value = mock_instance

                        result = runner.invoke(app, ["ask", "What is Vatuta?"])

                        assert result.exit_code == 0
                        assert "Vatuta is a Personal AI Assistant." in result.output
