from unittest.mock import MagicMock
from src.sources.source import Source

def test_source_initialization():
    mock_config = MagicMock()
    mock_config.id = "test_source"
    secrets = {}
    storage_path = "/app/test_storage"

    source = Source(mock_config, secrets, storage_path)

    assert source.config == mock_config
    assert source.secrets == secrets
    assert source.source_id == "test_source"
    assert str(source.storage_path) == storage_path
