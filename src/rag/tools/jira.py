"""Jira related tools."""

import logging
from typing import Any, List, Type

from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.rag.qdrant_manager import QdrantDocumentManager
from src.sources.jira_source import JiraSource
from src.sources.source import Source

logger = logging.getLogger(__name__)


class JiraTicketSchema(BaseModel):
    """Schema for Jira ticket lookup."""

    ticket_ids: List[str] = Field(
        ..., description="List of specific Jira ticket keys (e.g., ['PROJ-123', 'TEAM-456'])."
    )


class JiraTicketTool(BaseTool):
    """Tool to retrieve specific Jira tickets directly from the vector store."""

    name: str = "jira_ticket_lookup"
    description: str = (
        "Useful for retrieving specific Jira tickets by their keys. "
        "Use this when the user asks about specific ticket IDs."
    )
    args_schema: Type[BaseModel] = JiraTicketSchema
    handle_tool_error: bool = True
    sources: List[Source] = Field(default_factory=list, exclude=True)
    manager: Any = Field(default=None, exclude=True)

    def __init__(self, sources: List[Source], manager: QdrantDocumentManager, **kwargs: Any) -> None:
        """Initialize the JiraTicketTool with sources and document manager.

        Args:
            sources: List of sources to use for the tool.
            manager: Document manager to use for the tool.
            **kwargs: Additional keyword arguments to pass to the tool.
        """
        super().__init__(sources=sources, manager=manager, **kwargs)

    def _run(self, ticket_ids: List[str]) -> List[Document]:
        """Execute the lookup across all sources."""
        if not ticket_ids:
            return []

        aggregated_docs = []
        found_any = False

        # Iterate over sources to find ALL that can handle these IDs
        for source in self.sources:
            # We only check Jira sources
            if not isinstance(source, JiraSource):
                continue

            # We pass the list of IDs to the source
            f = source.get_specific_query(ticket_ids)
            if f:
                logger.info(f"Source {source.source_id} recognized tickets {ticket_ids}")
                found_any = True
                # Use QdrantDocumentManager to fetch docs with this filter
                # We use a high limit (or None if supported) to get all matches for this specific source
                docs = self.manager.get_documents(filter=f, limit=None)
                if docs:
                    aggregated_docs.extend(docs)

        if found_any:
            logger.info(f"Retrieved {len(aggregated_docs)} documents for Jira Tickets: {ticket_ids}")
            return aggregated_docs

        logger.warning(f"No source recognized ticket keys: {ticket_ids}")
        return []
