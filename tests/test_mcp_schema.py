"""Unit tests for MCPToolWrapper schema generation using json-schema-to-pydantic."""

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, ValidationError

from src.rag.tools.mcp import MCPToolWrapper


@pytest.fixture
def mock_server() -> MagicMock:
    """Fixture providing a mock MCPServer instance."""
    server = MagicMock()
    return server


@pytest.fixture
def mock_async_runner() -> MagicMock:
    """Fixture providing a mock AsyncLoopThread instance."""
    runner = MagicMock()
    return runner


def test_mcp_tool_wrapper_schema_generation(mock_server: MagicMock, mock_async_runner: MagicMock) -> None:
    """Test that MCPToolWrapper correctly generates a Pydantic model from JSON Schema."""
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"},
            "limit": {"type": "integer", "default": 10},
            "filters": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["query"],
    }

    tool = MCPToolWrapper(
        server=mock_server,
        async_runner=mock_async_runner,
        tool_name="search_tool",
        description="A search tool",
        input_schema=input_schema,
    )

    # Check that args_schema is a subclass of BaseModel
    assert issubclass(tool.args_schema, BaseModel)

    # Check the model name was generated safely
    assert tool.args_schema.__name__ == "MCP_search_tool_Schema"

    # Test valid instantiation
    valid_instance = tool.args_schema(query="test query")
    assert valid_instance.query == "test query"
    assert valid_instance.limit == 10

    # Test that invalid instantiation raises ValidationError
    with pytest.raises(ValidationError):
        tool.args_schema(limit="not an integer")

    with pytest.raises(ValidationError):
        # Missing required field 'query'
        tool.args_schema()


def test_mcp_tool_wrapper_complex_schema(mock_server: MagicMock, mock_async_runner: MagicMock) -> None:
    """Test that MCPToolWrapper can handle complex schemas like nested objects."""
    input_schema = {
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                "required": ["name"],
            }
        },
        "required": ["user"],
    }

    tool = MCPToolWrapper(
        server=mock_server,
        async_runner=mock_async_runner,
        tool_name="complex_tool",
        description="A complex tool",
        input_schema=input_schema,
    )

    # Test valid nested structure
    valid_instance = tool.args_schema(user={"name": "Alice", "age": 30})
    # user should be parsed as a nested model or dictionary depending on json-schema-to-pydantic
    assert valid_instance.user.name == "Alice"
    assert valid_instance.user.age == 30

    # Missing required inner field
    with pytest.raises(ValidationError):
        tool.args_schema(user={"age": 30})
