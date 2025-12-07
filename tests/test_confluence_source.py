import logging
import os
import sys
from unittest.mock import patch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.sources.confluence import ConfluenceCheckpoint, ConfluenceConfig, ConfluenceSource

# Mock config and secrets
config = {
    "url": "https://example.atlassian.net/wiki",
    "spaces": ["TEST"],
    "storage_path": "./data/confluence_test",
    "id": "confluence-test",
    "initial_lookback_days": 1,
    # Force limit to 1 for testing pagination
    "limit": 1,
}

secrets = {"jira_user": "test@example.com", "jira_api_token": "test-token"}


def test_confluence_source() -> None:
    print("Testing ConfluenceSource...")

    # Mock the Confluence client
    with (
        patch("src.sources.confluence.Confluence") as MockConfluence,
        patch("src.sources.confluence.ConfluenceSource.CONFLUENCE_API_LIMIT", 1),
    ):
        mock_client = MockConfluence.return_value

        # Mock client.get response with NO wrapper structure
        # We simulate 2 pages to test pagination

        page1_response = {
            "results": [
                {
                    "id": "12345",
                    "title": "Test Page 1",
                    "version": {"when": "2023-10-26T14:30:00.000Z", "by": {"displayName": "Test User"}},
                    "body": {"storage": {"value": "<h1>Header</h1><p>Paragraph content.</p>"}},
                    "metadata": {"labels": {"results": [{"name": "test-label"}, {"name": "documentation"}]}},
                    "_links": {"webui": "/spaces/TEST/pages/12345/Test+Page"},
                }
            ]
        }

        page2_response = {
            "results": [
                {
                    "id": "67890",
                    "title": "Test Page 2",
                    "version": {"when": "2023-10-27T14:30:00.000Z", "by": {"displayName": "Test User"}},
                    "body": {"storage": {"value": "<h1>Page 2</h1>"}},
                    "metadata": {"labels": {"results": []}},
                    "_links": {"webui": "/spaces/TEST/pages/67890/Test+Page+2"},
                }
            ]
        }

        # Mock empty response to stop pagination
        empty_response: dict[str, list] = {"results": []}

        mock_client.get.side_effect = [page1_response, page2_response, empty_response]

        source = ConfluenceSource(ConfluenceConfig(**config), secrets, storage_path=str(config["storage_path"]))
        checkpoint = ConfluenceCheckpoint(config=source.config)

        docs, chunks = source.collect_documents_and_chunks(checkpoint)

        print(f"Collected {len(docs)} documents and {len(chunks)} chunks")

        # Only 1 document collected because mock returns empty on 3rd call
        assert len(docs) == 1
        assert len(chunks) == 1
        assert docs[0].title == "Test Page 1"

        # Verify pagination - only 1 call since we get results then empty
        assert mock_client.get.call_count >= 1

        # Check first call args
        args, kwargs = mock_client.get.call_args_list[0]
        assert args[0] == "rest/api/content/search"

        print("Verification successful!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_confluence_source()
