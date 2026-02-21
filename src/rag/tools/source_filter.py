"""Source-filtering RAG agent tool."""

import logging
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field

from src.rag.tools.base import AgentTool
from src.rag.tools.utils import merge_filter

logger = logging.getLogger(__name__)


class SourceFilterSchema(BaseModel):
    """Input schema for source filtering."""

    source_types: Optional[List[str]] = Field(
        None, description="List of source types to filter by (e.g., ['jira', 'confluence'])."
    )
    source_ids: Optional[List[str]] = Field(
        None, description="List of specific source IDs to filter by (e.g., ['jira-main', 'confluence-docs'])."
    )


class SourceFilterTool(AgentTool):
    """Tool that creates a Qdrant source filter and applies it to the agent state."""

    name: str = "source_filter"
    description: str = (
        "Useful for narrowing down search results to specific source types or instances. "
        "Use this when the user asks to search specifically in 'Jira', 'Confluence', or specific connected instances."
    )
    args_schema: Type[BaseModel] = SourceFilterSchema
    handle_tool_error: bool = True

    def _run(
        self,
        source_types: Optional[List[str]] = None,
        source_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Build a Qdrant filter matching documents from the specified sources.

        Uses ``should`` (OR) semantics: documents match if they belong to *any*
        of the specified types or IDs.  Returns an empty dict when both parameters
        are ``None`` or empty.
        """
        if not source_types and not source_ids:
            return {}

        conditions: List[Dict[str, Any]] = []

        if source_types:
            conditions.append({"key": "metadata.source", "match": {"any": [t.lower() for t in source_types]}})

        if source_ids:
            conditions.append({"key": "metadata.source_instance_id", "match": {"any": source_ids}})

        return {"should": conditions}

    def apply_to_state(
        self,
        state: Dict[str, Any],
        source_types: Optional[List[str]] = None,
        source_ids: Optional[List[str]] = None,
    ) -> str:
        """Build a source filter and merge it into ``state["dynamic_query"]``.

        Args:
            state: Mutable agent state dict.
            source_types: Source type names to include (e.g. ``["jira"]``).
            source_ids: Explicit source instance IDs to include.

        Returns:
            Human-readable observation for the LLM trajectory.
        """
        qdrant_filter = self._run(source_types=source_types, source_ids=source_ids)
        if not qdrant_filter:
            return "No source filter applied."

        # NOTE: merge_filter extends 'should' lists, so multiple calls accumulate.
        # For conflicting filter semantics (e.g. combining must/should across tools),
        # a more sophisticated merge strategy may be needed in the future.
        merge_filter(state, qdrant_filter)

        applied = []
        if source_types:
            applied.append(f"types={source_types}")
        if source_ids:
            applied.append(f"ids={source_ids}")
        return f"Applied source filter: {', '.join(applied)}"
