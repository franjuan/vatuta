import pytest

@pytest.fixture
def temp_dir(tmpdir):
    return tmpdir

@pytest.fixture
def mock_config():
    return {
        "rag": {
            "model_id": "bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            "temperature": 0.2,
            "max_tokens": 800,
            "top_k": 4
        },
        "sources": {
            "slack": {},
            "jira": {},
            "confluence": {}
        }
    }
