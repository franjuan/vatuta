"""Jira-related RAG agent tools."""

import logging
from typing import Any, Dict, List, Type

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from src.rag.qdrant_manager import QdrantDocumentManager
from src.rag.tools.base import AgentTool
from src.sources.jira_source import JiraSource
from src.sources.source import Source

logger = logging.getLogger(__name__)


class JiraTicketSchema(BaseModel):
    """Input schema for Jira ticket lookup."""

    ticket_ids: List[str] = Field(
        ..., description="List of specific Jira ticket keys (e.g., ['PROJ-123', 'TEAM-456'])."
    )


class JiraTicketTool(AgentTool):
    """Tool that retrieves Jira tickets from the vector store and appends them to the agent state."""

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
        """Initialise the tool with the configured sources and document manager.

        Args:
            sources: All configured data sources; only JiraSource instances are used.
            manager: Qdrant document manager used to fetch matching documents.
            **kwargs: Additional keyword arguments forwarded to BaseTool.
        """
        super().__init__(sources=sources, manager=manager, **kwargs)

    def _run(self, ticket_ids: List[str]) -> List[Document]:
        """Fetch documents for the given Jira ticket keys across all Jira sources.

        Returns an empty list when no tickets are found or *ticket_ids* is empty.
        """
        if not ticket_ids:
            return []

        aggregated: List[Document] = []
        found_any = False

        for source in self.sources:
            if not isinstance(source, JiraSource):
                continue

            qdrant_filter = source.get_specific_query(ticket_ids)
            if qdrant_filter:
                logger.info(f"Source {source.source_id} recognised tickets {ticket_ids}")
                found_any = True
                docs = self.manager.get_documents(filter=qdrant_filter, limit=None)
                if docs:
                    aggregated.extend(docs)

        if found_any:
            logger.info(f"Retrieved {len(aggregated)} documents for Jira tickets: {ticket_ids}")
            return aggregated

        logger.warning(f"No source recognised ticket keys: {ticket_ids}")
        return []

    def apply_to_state(self, state: Dict[str, Any], ticket_ids: List[str]) -> str:
        """Fetch Jira tickets and append them to ``state["specific_docs"]``.

        Args:
            state: Mutable agent state dict.
            ticket_ids: Jira ticket keys to look up (e.g. ``["PROJ-123"]``).

        Returns:
            Human-readable observation for the LLM trajectory.
        """
        docs = self._run(ticket_ids)
        if docs:
            state.setdefault("specific_docs", [])
            state["specific_docs"].extend(docs)
            return f"Retrieved {len(docs)} Jira tickets."
        return "No Jira tickets found."
