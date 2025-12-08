from unittest.mock import MagicMock
from src.rag.engine import RAGState, format_docs, build_graph

def test_format_docs():
    mock_docs = [MagicMock(), MagicMock()]
    mock_docs[0].page_content = "This is the first document."
    mock_docs[1].page_content = "This is the second document."

    result = format_docs(mock_docs)

    expected_output = """This is the first document.

This is the second document."""
    assert result == expected_output

def test_build_graph():
    mock_document_manager = MagicMock()
    mock_rag_module = MagicMock()

    k = 5
    graph = build_graph(mock_document_manager, k, mock_rag_module)

    assert graph is not None
    assert "retrieve" in graph.nodes
    assert "generate" in graph.nodes
