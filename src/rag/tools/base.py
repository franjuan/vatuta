"""Base class for RAG agent tools that can apply their result to the agent state."""

import inspect
from abc import abstractmethod
from typing import Any, Callable, Dict

from langchain_core.tools import BaseTool


class AgentTool(BaseTool):
    """Extension of BaseTool that encapsulates both core logic and state mutation.

    Each concrete subclass must implement:
    - `_run(...)`: pure computation (builds a Qdrant filter, fetches docs, etc.).
    - `apply_to_state(state, ...)`: calls `_run`, merges the result into the
      agent state, and returns a human-readable string for the LLM trajectory.

    The agent owns the list of AgentTool instances and uses `as_agent_callable`
    to wrap them into plain callables that DSPy ReAct can inspect and invoke.
    """

    @abstractmethod
    def apply_to_state(self, state: Dict[str, Any], *args: Any, **kwargs: Any) -> str:
        """Execute the tool and mutate the agent state in-place.

        Args:
            state: The mutable AgentState dict from the current graph node.
            *args: Positional tool arguments (tool-specific).
            **kwargs: Keyword tool arguments (same names as `_run` parameters).

        Returns:
            A short, human-readable string describing what happened (shown to the
            LLM as the tool observation in the ReAct trajectory).
        """
        ...

    def as_agent_callable(self, state: Dict[str, Any]) -> Callable[..., str]:
        """Return a plain callable that closes over *state* for DSPy ReAct.

        The returned function has the same name, docstring, and parameter
        annotations as `apply_to_state` — minus the leading `state` argument —
        so that DSPy can introspect it and build the correct JSON schema for the
        LLM prompt.

        Args:
            state: The mutable AgentState dict to capture in the closure.

        Returns:
            A callable that DSPy ReAct can use as a tool.
        """
        # Collect apply_to_state parameters, skipping 'self' and 'state'.
        sig = inspect.signature(self.apply_to_state)
        tool_params = {name: param for name, param in sig.parameters.items() if name not in ("self", "state")}

        def _callable(**kwargs: Any) -> str:
            return self.apply_to_state(state, **kwargs)

        # Propagate metadata so DSPy can introspect them.
        _callable.__name__ = self.name
        _callable.__doc__ = self.description
        _callable.__annotations__ = {
            name: param.annotation
            for name, param in tool_params.items()
            if param.annotation is not inspect.Parameter.empty
        }
        # Attach a proper signature so inspect.signature() returns correct params.
        _callable.__signature__ = sig.replace(parameters=list(tool_params.values()))  # type: ignore[attr-defined]

        return _callable
