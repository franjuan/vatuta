import tempfile

from src.sources.confluence import ConfluenceConfig, ConfluenceSource


def test_confluence_recursion_fix() -> None:
    # Create configuration for source initialization
    config = ConfluenceConfig(id="test-source", url="https://example.com", spaces=["TEST"])

    # Mock secrets
    secrets = {"jira_user": "user", "jira_api_token": "token"}

    # Initialize source (storage_path is required but not used for this test)
    source = ConfluenceSource(config, secrets, storage_path=tempfile.gettempdir())

    # Create deeply nested HTML that causes RecursionError in markdownify
    depth = 2000
    deeply_nested_html = "<div>" * depth + "Content" + "</div>" * depth

    # Process the HTML
    # This should NOT raise RecursionError due to our fix
    result = source._html_to_markdown(deeply_nested_html)

    # Verify fallback to plain text
    assert result == "Content"


def test_confluence_normal_conversion() -> None:
    # Verify normal conversion still works
    config = ConfluenceConfig(id="test-source", url="https://example.com", spaces=["TEST"])
    secrets = {"jira_user": "user", "jira_api_token": "token"}
    source = ConfluenceSource(config, secrets, storage_path=tempfile.gettempdir())

    html = "<h1>Title</h1><p>Body</p>"
    result = source._html_to_markdown(html)

    assert "# Title" in result
    assert "Body" in result
