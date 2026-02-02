"""Base source interface for data collection.

Defines the abstract interface for all data sources.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar

from src.entities.manager import EntityManager
from src.models.documents import ChunkRecord, DocumentUnit
from src.models.source_config import BaseSourceConfig

from .checkpoint import Checkpoint

TConfig = TypeVar("TConfig", bound=BaseSourceConfig)


class Source(Generic[TConfig]):
    """Base class for data sources.

    A source is a collection of documents that are collected from a source system.
    It is used to collect the documents from the source system and to chunk the documents.
    There may be multiple instances of the same source type, but with different configuration.

    Parameters:
        config: A Pydantic model configuration for the source.
        secrets: A dictionary of secrets for the source.
        storage_path: Path to the directory where raw messages/checkpoints are stored.
        entity_manager: Optional entity manager for linking entities.
    """

    def __init__(
        self,
        config: TConfig,
        secrets: dict,
        storage_path: str,
        entity_manager: Optional[EntityManager] = None,
    ):
        """Initialize the source with configuration and storage path."""
        self.config = config
        self.secrets = secrets
        self.entity_manager = entity_manager

        # Identifier to disambiguate multiple instances of same source type
        # (e.g., Slack workspace/team id, Jira cloud id/tenant)
        # We expect 'id' to be present in the configuration model
        source_id: str = self.config.id
        if not source_id:
            raise ValueError("Source ID is not set in configuration")
        self.source_id = source_id

        # Required base directory where raw messages/checkpoints are stored
        if not storage_path:
            raise ValueError("storage_path must be provided")
        self.storage_path = Path(storage_path).expanduser().resolve()
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def collect_documents_and_chunks(
        self,
        checkpoint: Checkpoint,
        update_checkpoint: bool = True,
        use_cached_data: bool = True,
        filters: Optional[Dict[str, List[str]]] = None,
    ) -> Tuple[List[DocumentUnit], List[ChunkRecord]]:
        """Collect domain documents and their chunks in a single pass.

        Args:
            checkpoint: Marker indicating where the previous collection ended.
            update_checkpoint: Whether to persist checkpoint state after writing new data.
            use_cached_data: If True, replay cached JSONL.gz files newer than the checkpoint
                before issuing API calls.
            filters: Optional list of identifiers by filtering type to restrict processing.

        Returns:
            Tuple[List[DocumentUnit], List[ChunkRecord]]: A pair with:
                - documents: Logical source items (e.g., threads or windows) enriched
                  with metadata. `DocumentUnit.document_id` must be stable and unique.
                - chunks: Flat, single-level chunks derived from the documents. Each
                  `ChunkRecord.parent_document_id` must reference a `DocumentUnit.document_id`.

        Contract:
            - Implementations should make the operation idempotent and incremental w.r.t. `checkpoint`.
            - Chunks should default to level=0 (flat). Sources that support hierarchical
              chunking can set `parent_chunk_id` and `level>0`.
            - The method should not persist results; it should only return them to the caller.
        """
        raise NotImplementedError

    def collect_cached_documents_and_chunks(
        self,
        filters: Optional[Dict[str, list[str]]] = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> Tuple[List[DocumentUnit], List[ChunkRecord]]:
        """Rehydrate documents/chunks exclusively from cached data.

        Implementations must only read from the local storage cache and MUST NOT mutate
        or persist the checkpoint. Channel and date filters are optional.

        Args:
            filters: Optional list of identifiers by filtering type to restrict processing.
            date_from: Optional UTC datetime lower bound (inclusive).
            date_to: Optional UTC datetime upper bound (inclusive).
        """
        raise NotImplementedError

    def get_specific_query(self, document_ids: List[str]) -> Optional[Dict[str, Any]]:
        """Return a Qdrant filter query if any of the document IDs match resources in this source.

        Args:
            document_ids: List of specific document identifiers (e.g. Jira keys).

        Returns:
            Optional[Dict[str, Any]]: A Qdrant filter dictionary (e.g. {"must": [...]}) if a
            match is found, or None if no specific identifier is detected.
        """
        return None
