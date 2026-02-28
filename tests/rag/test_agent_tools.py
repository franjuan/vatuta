"""Unit tests for AgentTool base class, utils, and apply_to_state implementations."""

from typing import Any, Dict

from src.rag.tools.date import DateFilterTool
from src.rag.tools.source_filter import SourceFilterTool
from src.rag.tools.utils import merge_filter

# ---------------------------------------------------------------------------
# merge_filter utility
# ---------------------------------------------------------------------------


def test_merge_filter_empty_filter_is_noop() -> None:
    """merge_filter with an empty dict must not modify state."""
    state: Dict[str, Any] = {}
    merge_filter(state, {})
    assert state == {}


def test_merge_filter_creates_dynamic_query_if_absent() -> None:
    """merge_filter creates state['dynamic_query'] when it does not exist yet."""
    state: Dict[str, Any] = {}
    merge_filter(state, {"must": [{"key": "foo", "range": {"gte": "2024-01-01"}}]})
    assert "dynamic_query" in state
    assert len(state["dynamic_query"]["must"]) == 1


def test_merge_filter_extends_existing_must_clause() -> None:
    """merge_filter appends to an existing 'must' list instead of replacing it."""
    state: Dict[str, Any] = {"dynamic_query": {"must": [{"key": "existing"}]}}
    merge_filter(state, {"must": [{"key": "new"}]})
    assert len(state["dynamic_query"]["must"]) == 2


def test_merge_filter_handles_should_clause() -> None:
    """merge_filter correctly extends 'should' clauses."""
    state: Dict[str, Any] = {}
    merge_filter(state, {"should": [{"key": "metadata.source", "match": {"any": ["jira"]}}]})
    assert len(state["dynamic_query"]["should"]) == 1


def test_merge_filter_accumulates_multiple_calls() -> None:
    """Multiple merge_filter calls accumulate all conditions."""
    state: Dict[str, Any] = {}
    merge_filter(state, {"must": [{"key": "date"}]})
    merge_filter(state, {"should": [{"key": "source_type"}]})
    assert len(state["dynamic_query"]["must"]) == 1
    assert len(state["dynamic_query"]["should"]) == 1


# ---------------------------------------------------------------------------
# DateFilterTool.apply_to_state
# ---------------------------------------------------------------------------


def test_date_filter_apply_no_dates() -> None:
    """apply_to_state returns a 'no filter' message when no dates are given."""
    state: Dict[str, Any] = {}
    tool = DateFilterTool()
    result = tool.apply_to_state(state)
    assert result == "No date filter applied."
    assert state == {}


def test_date_filter_apply_start_only() -> None:
    """apply_to_state merges a 'gte' filter when only start_date is supplied."""
    state: Dict[str, Any] = {}
    tool = DateFilterTool()
    result = tool.apply_to_state(state, start_date="2024-01-01")
    assert "Applied date filter" in result
    must = state["dynamic_query"]["must"]
    assert len(must) == 1
    assert must[0]["range"]["gte"] == "2024-01-01"
    assert "lte" not in must[0]["range"]


def test_date_filter_apply_both_dates() -> None:
    """apply_to_state merges both 'gte' and 'lte' when both dates are supplied."""
    state: Dict[str, Any] = {}
    DateFilterTool().apply_to_state(state, start_date="2024-01-01", end_date="2024-01-31")
    must = state["dynamic_query"]["must"]
    assert must[0]["range"]["gte"] == "2024-01-01"
    assert must[0]["range"]["lte"] == "2024-01-31"


def test_date_filter_apply_accumulates() -> None:
    """Two apply_to_state calls accumulate two 'must' conditions."""
    state: Dict[str, Any] = {}
    tool = DateFilterTool()
    tool.apply_to_state(state, start_date="2024-01-01", end_date="2024-01-31")
    tool.apply_to_state(state, start_date="2024-06-01", end_date="2024-06-30")
    assert len(state["dynamic_query"]["must"]) == 2


# ---------------------------------------------------------------------------
# SourceFilterTool.apply_to_state
# ---------------------------------------------------------------------------


def test_source_filter_apply_no_params() -> None:
    """apply_to_state returns a 'no filter' message when nothing is specified."""
    state: Dict[str, Any] = {}
    result = SourceFilterTool().apply_to_state(state)
    assert result == "No source filter applied."
    assert state == {}


def test_source_filter_apply_types() -> None:
    """apply_to_state merges a source-type 'should' condition."""
    state: Dict[str, Any] = {}
    SourceFilterTool().apply_to_state(state, source_types=["jira"])
    should = state["dynamic_query"]["should"]
    assert any(c["key"] == "metadata.source" for c in should)


def test_source_filter_apply_ids() -> None:
    """apply_to_state merges a source-id 'should' condition."""
    state: Dict[str, Any] = {}
    SourceFilterTool().apply_to_state(state, source_ids=["jira-main"])
    should = state["dynamic_query"]["should"]
    assert any(c["key"] == "metadata.source_instance_id" for c in should)


# ---------------------------------------------------------------------------
# AgentTool.as_agent_callable
# ---------------------------------------------------------------------------


def test_as_agent_callable_name_and_doc() -> None:
    """as_agent_callable exposes the tool's name and description."""
    tool = DateFilterTool()
    state: Dict[str, Any] = {}
    fn = tool.as_agent_callable(state)
    assert fn.__name__ == "date_filter"
    assert fn.__doc__ is not None and len(fn.__doc__) > 0


def test_as_agent_callable_invokes_apply_to_state() -> None:
    """as_agent_callable correctly delegates to apply_to_state."""
    state: Dict[str, Any] = {}
    fn = DateFilterTool().as_agent_callable(state)
    result = fn(start_date="2024-03-01", end_date="2024-03-31")
    assert "Applied date filter" in result
    assert "must" in state["dynamic_query"]


def test_as_agent_callable_state_excludes_state_param() -> None:
    """The callable signature must not expose the 'state' parameter to DSPy."""
    import inspect

    fn = DateFilterTool().as_agent_callable({})
    params = inspect.signature(fn).parameters
    assert "state" not in params, "'state' must not be visible in the callable signature"
