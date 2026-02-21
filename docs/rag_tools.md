# RAG Tools — Integration Guide

This document describes how the RAG agent's tool system works and how to add new tools.

## Overview

Each routing tool encapsulates **two responsibilities**:

| Layer | Method | Responsibility |
| --- | --- | --- |
| Pure logic | `_run(...)` | Builds a Qdrant filter dict or fetches documents — no side effects |
| State mutation | `apply_to_state(state, ...)` | Calls `_run`, merges the result into `AgentState`, returns an LLM-readable string |

This design means the agent's `route()` node stays stable regardless of how many tools are added.

## Architecture

```text
AgentTool (abstract, src/rag/tools/base.py)
├── _run(...)             ← pure computation
├── apply_to_state(...)   ← calls _run + mutates state
└── as_agent_callable(state) → Callable   ← wraps apply_to_state for dspy.ReAct

Concrete tools
├── DateFilterTool        (src/rag/tools/date.py)
├── SourceFilterTool      (src/rag/tools/source_filter.py)
└── JiraTicketTool        (src/rag/tools/jira.py)

Shared utilities
└── merge_filter          (src/rag/tools/utils.py)
```

### How `as_agent_callable` works

DSPy's `ReAct` module needs a list of plain Python callables to build its prompt.
`as_agent_callable(state)` returns a closure over the current graph state:

```python
# Inside route():
tools = [tool.as_agent_callable(state) for tool in self._tools]
router_agent = dspy.ReAct(self.router_signature, tools=tools, max_iters=5)
```

The returned callable:

- Has the **same name** as `AgentTool.name` → appears correctly in the ReAct prompt.
- Has the **same docstring** as `AgentTool.description` → used as tool description.
- Has the **same parameter signature** as `apply_to_state` minus `(self, state)` →
  DSPy introspects this to build the JSON schema for the LLM.
- The `state` argument is **never exposed** to the LLM — it is captured in the closure.

### Qdrant filter accumulation

Tools that produce Qdrant filters call `merge_filter(state, new_filter)` from
`src/rag/tools/utils.py`. This utility extends `state["dynamic_query"]` in-place:
`must` and `should` clause lists are **appended to**, never replaced.
This allows multiple tool calls in a single routing turn to accumulate their
filters correctly.

> **Note**: the current merge strategy is additive. Complex scenarios that require
> combining `must`/`should` across different tools (e.g. "in Jira AND last week")
> work correctly because date filters produce a `must` condition and source filters
> produce a `should` condition. If future tools need more sophisticated merging,
> extend `merge_filter` or override `apply_to_state`.

## Existing Tools

### `DateFilterTool`

| | |
| --- | --- |
| **File** | `src/rag/tools/date.py` |
| **Trigger** | User implies a time range ("last week", "since December", "in 2024") |
| **`_run` output** | `{"must": [{"key": "metadata.source_updated_at", "range": {...}}]}` |
| **State effect** | Extends `state["dynamic_query"]["must"]` |

### `SourceFilterTool`

| | |
| --- | --- |
| **File** | `src/rag/tools/source_filter.py` |
| **Trigger** | User asks to search in a specific source ("in Jira", "in the Confluence space") |
| **`_run` output** | `{"should": [{"key": "metadata.source", "match": {"any": [...]}}]}` |
| **State effect** | Extends `state["dynamic_query"]["should"]` |
| **Logic** | OR semantics: documents from *any* of the listed types or IDs match |

### `JiraTicketTool`

| | |
| --- | --- |
| **File** | `src/rag/tools/jira.py` |
| **Trigger** | User mentions explicit Jira ticket IDs (e.g. `PROJ-123`) |
| **`_run` output** | `List[Document]` fetched from Qdrant via exact match |
| **State effect** | Extends `state["specific_docs"]` |
| **Dependencies** | Requires `sources: List[Source]` and `manager: QdrantDocumentManager` |

## Adding a New Tool

1. **Create** a new file `src/rag/tools/my_tool.py`.
2. **Inherit** from `AgentTool`:

```python
from src.rag.tools.base import AgentTool
from src.rag.tools.utils import merge_filter

class MyTool(AgentTool):
    name: str = "my_tool"
    description: str = "Describe when the LLM should call this tool."

    def _run(self, param: str) -> dict:
        """Pure logic — build and return the Qdrant filter (or fetch data)."""
        return {"must": [{"key": "metadata.my_field", "match": {"value": param}}]}

    def apply_to_state(self, state: dict, param: str) -> str:
        """Apply the tool result to the agent state."""
        result = self._run(param)
        if not result:
            return "No filter applied."
        merge_filter(state, result)
        return f"Applied filter for param={param}"
```

1. **Register** it in `RAGAgent.__init__`:

```python
self._tools: List[AgentTool] = [
    DateFilterTool(),
    JiraTicketTool(sources=self.sources, manager=self.doc_manager),
    SourceFilterTool(),
    MyTool(),   # ← add here
]
```

1. **Export** it from `src/rag/tools/__init__.py`.
2. **Test** it in `tests/rag/test_agent_tools.py` following the existing pattern.

No changes to `route()` or the graph structure are required.

## Testing Tools in Isolation

All tools can be tested without a running LLM or Qdrant instance:

```python
# Test pure filter logic
def test_my_tool_run():
    assert MyTool()._run("value") == {"must": [...]}

# Test state mutation
def test_my_tool_apply_to_state():
    state = {}
    MyTool().apply_to_state(state, param="value")
    assert "dynamic_query" in state

# Test the callable wrapper
def test_my_tool_callable():
    fn = MyTool().as_agent_callable(state={})
    assert fn.__name__ == "my_tool"
    assert "state" not in inspect.signature(fn).parameters
```
