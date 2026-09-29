"""Configuration models for Vatuta application.

This module defines the configuration structure for RAG settings and data sources.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from src.mcp.config import MCPContainerConfig
from src.sources.confluence import ConfluenceConfig
from src.sources.jira_source import JiraConfig
from src.sources.slack import SlackConfig


class LLMBackendConfig(BaseModel):
    """Configuration definition for a single LLM backend."""

    model: str = Field(
        ...,
        description="LiteLLM model identifier in 'provider/model_name' format (e.g. 'gemini/gemini-2.5-flash')",
    )
    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for text generation",
    )
    max_tokens: Optional[int] = Field(
        default=800,
        gt=0,
        description="Maximum token generation limit",
    )
    top_p: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling probability threshold",
    )
    api_base: Optional[str] = Field(
        default=None,
        description="Custom API base URL endpoint if proxying or using enterprise endpoints",
    )
    extra_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific passthrough kwargs for LiteLLM completion calls",
    )

    @model_validator(mode="before")
    @classmethod
    def handle_legacy_model_id(cls, data: Any) -> Any:
        """Support legacy model_id parameter during deserialization."""
        if isinstance(data, dict):
            if "model" not in data and "model_id" in data:
                data = dict(data)
                data["model"] = data.pop("model_id")
        return data

    @field_validator("model")
    @classmethod
    def validate_model_format(cls, v: str) -> str:
        """Validate that model string conforms to provider/model_name format."""
        parts = v.strip().split("/")
        if len(parts) < 2 or not parts[0] or not parts[1]:
            raise ValueError(
                f"Model identifier '{v}' must be in 'provider/model_name' format (e.g., 'gemini/gemini-2.5-flash')"
            )
        return v.strip()


# Backwards compatibility alias
LLMConfig = LLMBackendConfig


class RagConfig(BaseModel):
    """Configuration for RAG (Retrieval-Augmented Generation) system."""

    llm_backends: Dict[str, LLMBackendConfig] = Field(
        ...,
        min_length=1,
        description="Registry of configured LLM backends",
    )
    router_backend: str = Field(
        ...,
        description="Backend ID from llm_backends bound to query routing",
    )
    generator_backend: str = Field(
        ...,
        description="Backend ID from llm_backends bound to answer synthesis",
    )

    @model_validator(mode="after")
    def validate_backend_bindings(self) -> "RagConfig":
        """Validate that router_backend and generator_backend exist in llm_backends."""
        available = list(self.llm_backends.keys())
        if self.router_backend not in self.llm_backends:
            raise ValueError(
                f"router_backend '{self.router_backend}' not found in llm_backends. Available: {', '.join(available)}"
            )
        if self.generator_backend not in self.llm_backends:
            raise ValueError(
                f"generator_backend '{self.generator_backend}' not found in llm_backends. Available: {', '.join(available)}"
            )
        return self


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
        ...,
        description="HuggingFace embeddings model",
    )
    embeddings_query_prefix: Optional[str] = Field(
        default=None,
        description="Prefix to prepend to queries during semantic search",
    )
    embeddings_document_prefix: Optional[str] = Field(
        default=None,
        description="Prefix to prepend to documents during indexing",
    )
    embeddings_normalize: bool = Field(
        default=False,
        description="Whether to normalize dense embeddings (L2 normalization)",
    )
    dense_vector_name: str = Field(
        default="dense",
        description="Name of the dense vector field in Qdrant",
    )
    sparse_vector_name: str = Field(
        default="sparse",
        description="Name of the sparse vector field in Qdrant",
    )
    sparse_embeddings_model: str = Field(
        ...,
        description="FastEmbed model for sparse embeddings",
    )


class VatutaConfig(BaseModel):
    """Main configuration for Vatuta application."""

    rag: RagConfig
    qdrant: QdrantConfig
    sources: SourcesConfig
    entities_manager: EntityManagerConfig = Field(default_factory=EntityManagerConfig)
    mcp_servers: Dict[str, MCPContainerConfig] = Field(default_factory=dict)


class ConfigLoader:
    """Utility class for loading configuration from YAML files."""

    @staticmethod
    def load(path: str) -> VatutaConfig:
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            VatutaConfig: Loaded configuration object.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Configuration file not found at: {path}")

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

        # Inject names into mcp server configs if they are missing
        if "mcp_servers" in raw_data:
            servers = raw_data["mcp_servers"]
            for server_name, server_config in servers.items():
                if isinstance(server_config, dict):
                    # Inject the dictionary key as 'name' if not provided
                    if "name" not in server_config:
                        server_config["name"] = server_name

        return VatutaConfig(**raw_data)
