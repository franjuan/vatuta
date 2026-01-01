"""Configuration models for Vatuta application.

This module defines the configuration structure for RAG settings and data sources.
"""

from pathlib import Path
from typing import Dict

import yaml
from pydantic import BaseModel, Field

from src.sources.confluence import ConfluenceConfig
from src.sources.jira_source import JiraConfig
from src.sources.slack import SlackConfig


class LLMConfig(BaseModel):
    """Configuration for a specific LLM backend."""

    model_id: str
    temperature: float = 0.2
    max_tokens: int = 800
    top_k: int = 4


class RagConfig(BaseModel):
    """Configuration for RAG (Retrieval-Augmented Generation) system."""

    llm_backend: Dict[str, LLMConfig] = Field(default_factory=dict)


class SourcesConfig(BaseModel):
    """Configuration for all data sources (Slack, Jira, Confluence)."""

    slack: Dict[str, SlackConfig] = Field(default_factory=dict)
    jira: Dict[str, JiraConfig] = Field(default_factory=dict)
    confluence: Dict[str, ConfluenceConfig] = Field(default_factory=dict)


class EntityManagerConfig(BaseModel):
    """Configuration for entity manager."""

    storage_path: str = Field(default="data/entities.json", description="Path to global entities storage file")


class QdrantConfig(BaseModel):
    """Configuration for Qdrant vector database."""

    url: str = Field(default="http://localhost:6333", description="Qdrant server URL")
    collection_name: str = Field(default="vatuta_documents", description="Collection name for documents")
    embeddings_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace embeddings model",
    )


class VatutaConfig(BaseModel):
    """Main configuration for Vatuta application."""

    rag: RagConfig = Field(default_factory=RagConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    sources: SourcesConfig = Field(default_factory=SourcesConfig)
    entities_manager: EntityManagerConfig = Field(default_factory=EntityManagerConfig)


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
