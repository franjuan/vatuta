"""Configuration models for Vatuta application.

This module defines the configuration structure for RAG settings and data sources.
"""

from pathlib import Path
from typing import Dict

import yaml
from pydantic import BaseModel, Field

from src.sources.confluence import ConfluenceConfig
from src.sources.jira import JiraConfig
from src.sources.slack import SlackConfig


class RagConfig(BaseModel):
    """Configuration for RAG (Retrieval-Augmented Generation) system."""

    model_id: str = "bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0"
    temperature: float = 0.2
    max_tokens: int = 800
    top_k: int = 4


class SourcesConfig(BaseModel):
    """Configuration for all data sources (Slack, Jira, Confluence)."""

    slack: Dict[str, SlackConfig] = Field(default_factory=dict)
    jira: Dict[str, JiraConfig] = Field(default_factory=dict)
    confluence: Dict[str, ConfluenceConfig] = Field(default_factory=dict)


class VatutaConfig(BaseModel):
    """Main configuration for Vatuta application."""

    rag: RagConfig = Field(default_factory=RagConfig)
    sources: SourcesConfig = Field(default_factory=SourcesConfig)


class ConfigLoader:
    """Utility class for loading configuration from YAML files."""

    @staticmethod
    def load(path: str) -> VatutaConfig:
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            VatutaConfig: Loaded configuration object.
        """
        p = Path(path)
        if not p.exists():
            return VatutaConfig()

        with open(p, "r") as f:
            raw_data = yaml.safe_load(f) or {}

        # Inject IDs into source configs if they are missing
        if "sources" in raw_data:
            sources = raw_data["sources"]
            for source_type in ["slack", "jira", "confluence"]:
                if source_type in sources:
                    for source_id, source_config in sources[source_type].items():
                        if isinstance(source_config, dict):
                            # Inject the dictionary key as 'id' if not provided
                            if "id" not in source_config:
                                source_config["id"] = source_id

        return VatutaConfig(**raw_data)
