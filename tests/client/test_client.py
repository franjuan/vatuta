from typer.testing import CliRunner
from src.client.client import app
from unittest.mock import patch

runner = CliRunner()

@patch('src.client.client.DocumentManager')
def test_app_reset_command(mock_document_manager):
    result = runner.invoke(app, ["reset"], input="y\n")
    assert result.exit_code == 0
    assert "Knowledge Base cleared successfully" in result.stdout

def test_app_load_command():
    result = runner.invoke(app, ["load"])
    assert result.exit_code == 0
    assert "No enabled sources matches the filters" in result.stdout

def test_app_update_command():
    result = runner.invoke(app, ["update"])
    assert result.exit_code == 0
    assert "No enabled sources matches the filters" in result.stdout

@patch('src.client.client.DocumentManager')
@patch('src.client.client.build_graph')
@patch('src.client.client.configure_dspy_lm')
def test_app_ask_command_success(mock_configure_dspy_lm, mock_build_graph, mock_document_manager):
    # Mock the RAG graph and its invoke method
    mock_graph = mock_build_graph.return_value
    mock_graph.invoke.return_value = {"answer": "This is a mock answer."}

    # Run the ask command
    result = runner.invoke(app, ["ask", "What is your name?"])

    # Assert that the command runs successfully and returns the mock answer
    assert result.exit_code == 0
    assert "This is a mock answer." in result.stdout
