import tempfile
from unittest.mock import patch

import pytest

from src.sources.confluence import ConfluenceConfig, ConfluenceSource
from src.sources.jira_source import JiraConfig, JiraSource


@pytest.fixture
def mock_jira_config() -> JiraConfig:
    return JiraConfig(
        url="https://jira.example.com",
        projects=["PROJ"],
        id="jira-test",
    )


@pytest.fixture
def mock_confluence_config() -> ConfluenceConfig:
    return ConfluenceConfig(
        url="https://confluence.example.com",
        spaces=["SPACE"],
        id="confluence-test",
    )


def test_jira_offline_init(mock_jira_config: JiraConfig) -> None:
    with patch("src.sources.jira_source.JIRA") as MockJira:
        mock_client = MockJira.return_value

        # Configure search to return empty issues so loop terminates
        mock_client.enhanced_search_issues.return_value = {
            "issues": [],
            "names": {},
            "schema": {},
            "nextPageToken": None,
        }

        # Init should NOT call myslf()
        source = JiraSource(
            mock_jira_config, secrets={"jira_user": "u", "jira_api_token": "t"}, storage_path=tempfile.gettempdir()
        )

        # Verify JIRA was initialized with deferred validation options
        mock_client = MockJira.return_value
        MockJira.assert_called_with(
            server=mock_jira_config.url, basic_auth=("u", "t"), validate=False, get_server_info=False
        )
        mock_client.myself.assert_not_called()

        # Verify call in collect_documents_and_chunks
        source.collect_documents_and_chunks(checkpoint=None)  # type: ignore[arg-type]
        mock_client.myself.assert_called_once()


def test_jira_cached_no_validation(mock_jira_config: JiraConfig) -> None:
    with patch("src.sources.jira_source.JIRA") as MockJira:
        mock_client = MockJira.return_value
        source = JiraSource(
            mock_jira_config, secrets={"jira_user": "u", "jira_api_token": "t"}, storage_path=tempfile.gettempdir()
        )

        # Verify NO call in collect_cached_documents_and_chunks
        source.collect_cached_documents_and_chunks(filters=None)
        mock_client.myself.assert_not_called()


def test_confluence_offline_init(mock_confluence_config: ConfluenceConfig) -> None:
    with patch("src.sources.confluence.Confluence") as MockConf:
        mock_client = MockConf.return_value

        # Configure get to return empty results to break loop
        mock_client.get.return_value = {"results": [], "_links": {}}

        # Init should NOT call API
        source = ConfluenceSource(
            mock_confluence_config,
            secrets={"jira_user": "u", "jira_api_token": "t"},
            storage_path=tempfile.gettempdir(),
        )
        mock_client.get.assert_not_called()

        # Verify call in collect_documents_and_chunks
        source.collect_documents_and_chunks(checkpoint=None)  # type: ignore[arg-type]

        # Check if validation call was made
        args_list = mock_client.get.call_args_list
        assert any(args[0][0] == "rest/api/user/current" for args in args_list), "Validation call not found"


def test_confluence_cached_no_validation(mock_confluence_config: ConfluenceConfig) -> None:
    with patch("src.sources.confluence.Confluence") as MockConf:
        mock_client = MockConf.return_value
        source = ConfluenceSource(
            mock_confluence_config,
            secrets={"jira_user": "u", "jira_api_token": "t"},
            storage_path=tempfile.gettempdir(),
        )

        # Verify NO call in collect_cached_documents_and_chunks
        source.collect_cached_documents_and_chunks(filters=None)
        mock_client.get.assert_not_called()
