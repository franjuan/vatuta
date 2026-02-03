"""Tool for filtering search results by source."""

import logging
from typing import Any, Dict, List, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SourceFilterSchema(BaseModel):
    """Schema for source filtering parameters."""

    source_type: Optional[str] = Field(
        None, description="The type of source to filter by (e.g., 'jira', 'confluence')."
    )
    source_id: Optional[str] = Field(
        None, description="The specific source ID to filter by (e.g., 'jira-main', 'confluence-docs')."
    )


class SourceFilterTool(BaseTool):
    """Tool to create source filters for vector search."""

    name: str = "source_filter"
    description: str = (
        "Useful for narrowing down search results to a specific source type or instance. "
        "Use this when the user asks to search specifically in 'Jira', 'Confluence', or a specific connected instance."
    )
    args_schema: Type[BaseModel] = SourceFilterSchema
    handle_tool_error: bool = True

    def _run(self, source_type: Optional[str] = None, source_id: Optional[str] = None) -> Dict[str, Any]:
        """Convert source identifiers to Qdrant filter."""
        if not source_type and not source_id:
            return {}

        must_conditions: List[Dict[str, Any]] = []

        if source_type:
            must_conditions.append({"key": "metadata.source", "match": {"value": source_type.lower()}})

        if source_id:
            must_conditions.append({"key": "metadata.source_instance_id", "match": {"value": source_id}})

        return {"must": must_conditions}
