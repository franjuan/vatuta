"""Tests for the SourceFilterTool."""

from src.rag.tools.source_filter import SourceFilterTool


def test_source_filter_empty() -> None:
    """Test that empty input returns an empty dict."""
    tool = SourceFilterTool()
    result = tool._run()
    assert result == {}


def test_source_filter_single_type() -> None:
    """Test filtering by a single source type."""
    tool = SourceFilterTool()
    result = tool._run(source_types=["jira"])
    assert result == {"should": [{"key": "metadata.source", "match": {"any": ["jira"]}}]}


def test_source_filter_multiple_types() -> None:
    """Test filtering by multiple source types."""
    tool = SourceFilterTool()
    result = tool._run(source_types=["jira", "confluence"])
    # The order in "any" list depends on input order
    assert result["should"][0]["key"] == "metadata.source"
    assert set(result["should"][0]["match"]["any"]) == {"jira", "confluence"}


def test_source_filter_single_id() -> None:
    """Test filtering by a single source ID."""
    tool = SourceFilterTool()
    result = tool._run(source_ids=["jira-123"])
    assert result == {"should": [{"key": "metadata.source_instance_id", "match": {"any": ["jira-123"]}}]}


def test_source_filter_multiple_ids() -> None:
    """Test filtering by multiple source IDs."""
    tool = SourceFilterTool()
    result = tool._run(source_ids=["jira-123", "confluence-456"])
    assert result["should"][0]["key"] == "metadata.source_instance_id"
    assert set(result["should"][0]["match"]["any"]) == {"jira-123", "confluence-456"}


def test_source_filter_types_and_ids() -> None:
    """Test filtering by both source types and IDs."""
    tool = SourceFilterTool()
    result = tool._run(source_types=["jira"], source_ids=["confluence-456"])

    assert len(result["should"]) == 2

    # Check for type filter
    type_filter = next(f for f in result["should"] if f["key"] == "metadata.source")
    assert type_filter["match"]["any"] == ["jira"]

    # Check for ID filter
    id_filter = next(f for f in result["should"] if f["key"] == "metadata.source_instance_id")
    assert id_filter["match"]["any"] == ["confluence-456"]
