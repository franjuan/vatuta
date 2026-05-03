"""Unit tests for the max_tokens override feature."""

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from src.client.client import app
from src.models.config import ConfigLoader


def test_ask_max_tokens_override() -> None:
    """Test that max_tokens overrides the generator backend configuration."""
    runner = CliRunner()

    # Mocking ConfigLoader.load and RAGAgent.run
    mock_config = MagicMock()
    mock_config.rag.generator_backend = "gemini"
    mock_gemini = MagicMock()
    mock_gemini.max_tokens = 800
    mock_config.rag.llm_backends = {"gemini": mock_gemini}
    mock_config.entities_manager.storage_path = "data/entities.json"

    with patch.object(ConfigLoader, "load", return_value=mock_config):
        with patch("src.client.client.QdrantDocumentManager"):
            with patch("src.client.client._get_enabled_sources", return_value=[]):
                with patch("src.rag.agent.RAGAgent") as MockAgent:
                    mock_agent_instance = MagicMock()
                    mock_agent_instance.run.return_value = {
                        "answer": "Answer with overridden tokens",
                        "router_cot": {},
                        "generator_cot": "",
                        "routing_summary": "",
                    }
                    MockAgent.return_value = mock_agent_instance

                    result = runner.invoke(app, ["ask", "What is Vatuta?", "--max-tokens", "1000"])

                    assert result.exit_code == 0
                    assert mock_gemini.max_tokens == 1000
