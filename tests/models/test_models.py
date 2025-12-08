from src.models.config import ConfigLoader, VatutaConfig

def test_config_loader(temp_dir, mock_config):
    config_path = temp_dir.join("config.yml")
    with open(config_path, "w") as f:
        f.write("""
        rag:
            model_id: "bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0"
            temperature: 0.2
            max_tokens: 800
            top_k: 4
        sources:
            slack: {}
            jira: {}
            confluence: {}
        """)

    config = ConfigLoader.load(str(config_path))
    assert isinstance(config, VatutaConfig)
    assert config.rag.model_id == "bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0"
