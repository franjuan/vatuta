"""Model Context Protocol (MCP) tool wrappers for RAG Agent."""

import logging
from typing import Any, Dict

from json_schema_to_pydantic import create_model as create_model_from_schema
from langchain_core.documents import Document

from src.mcp.server import MCPServer
from src.rag.tools.base import AgentTool
from src.utils.async_runner import AsyncLoopThread

logger = logging.getLogger(__name__)


class MCPToolWrapper(AgentTool):
    """Wrapper for MCP Tools to be used in LangGraph RAG Agent."""

    # Using private attributes so Pydantic doesn't try to validate them
    _server: MCPServer
    _async_runner: AsyncLoopThread
    _tool_name: str

    def __init__(
        self,
        server: MCPServer,
        async_runner: AsyncLoopThread,
        tool_name: str,
        description: str,
        input_schema: Dict[str, Any],
    ):
        """Initialize the MCP Tool Wrapper.

        Args:
            server: The MCPServer instance.
            async_runner: The AsyncLoopThread to run async calls.
            tool_name: The name of the tool in MCP.
            description: Description of the tool.
            input_schema: JSON schema of the tool arguments.
        """
        # Create args_schema dynamically based on input_schema
        # Ensure name is a valid Python identifier
        safe_name = "".join(c if c.isalnum() else "_" for c in tool_name)

        # Inject the title so the generated Pydantic model has a clear class name
        input_schema_copy = dict(input_schema)
        input_schema_copy["title"] = f"MCP_{safe_name}_Schema"

        schema_model = create_model_from_schema(input_schema_copy)

        super().__init__(name=tool_name, description=description, args_schema=schema_model)
        self._server = server
        self._async_runner = async_runner
        self._tool_name = tool_name

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the tool."""
        raise NotImplementedError("Use apply_to_state instead.")

    def apply_to_state(self, state: Dict[str, Any], *args: Any, **kwargs: Any) -> str:
        """Execute the tool on the MCP server and return a description of the result.

        Args:
            state: The mutable AgentState dict from the current graph node.
            *args: Positional tool arguments.
            **kwargs: Tool arguments.

        Returns:
            str: Result of the MCP tool execution.
        """
        try:
            # Execute async MCP call in the background thread
            result = self._async_runner.run_coroutine(self._server.call_tool(self._tool_name, kwargs))

            # Format the output for the LLM
            content_pieces = []
            for item in getattr(result, "content", []):
                if getattr(item, "type", "") == "text":
                    content_pieces.append(getattr(item, "text", ""))
                else:
                    content_pieces.append(str(item))

            output_text = "\n".join(content_pieces)
            if not output_text:
                output_text = str(result)

            if "specific_docs" not in state:
                state["specific_docs"] = []

            state["specific_docs"].append(
                Document(
                    page_content=output_text,
                    metadata={"source": f"mcp_{self._tool_name}", "title": f"MCP Tool Output: {self._tool_name}"},
                )
            )

            return f"Tool '{self._tool_name}' returned:\n{output_text}"
        except Exception as e:
            logger.error("Error executing tool '%s': %s", self._tool_name, e, exc_info=True)
            return f"Error executing tool '{self._tool_name}': {e}"
