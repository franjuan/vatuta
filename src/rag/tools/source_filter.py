"""Tool for filtering search results by source."""

import logging
from typing import Any, Dict, List, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SourceFilterSchema(BaseModel):
    """Schema for source filtering parameters."""

    source_types: Optional[List[str]] = Field(
        None, description="List of source types to filter by (e.g., ['jira', 'confluence'])."
    )
    source_ids: Optional[List[str]] = Field(
        None, description="List of specific source IDs to filter by (e.g., ['jira-main', 'confluence-docs'])."
    )


class SourceFilterTool(BaseTool):
    """Tool to create source filters for vector search."""

    name: str = "source_filter"
    description: str = (
        "Useful for narrowing down search results to specific source types or instances. "
        "Use this when the user asks to search specifically in 'Jira', 'Confluence', or specific connected instances."
    )
    args_schema: Type[BaseModel] = SourceFilterSchema
    handle_tool_error: bool = True

    def _run(self, source_types: Optional[List[str]] = None, source_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Convert source identifiers to Qdrant filter."""
        if not source_types and not source_ids:
            return {}

        must_conditions: List[Dict[str, Any]] = []

        if source_types:
            must_conditions.append({"key": "metadata.source", "match": {"any": [t.lower() for t in source_types]}})

        if source_ids:
            must_conditions.append({"key": "metadata.source_instance_id", "match": {"any": source_ids}})

        return {"should": must_conditions}
