"""Shared utilities for RAG agent tools."""

from typing import Any, Dict


def merge_filter(state: Dict[str, Any], new_filter: Dict[str, Any]) -> None:
    """Merge a new Qdrant filter dict into ``state["dynamic_query"]`` in-place.

    Both ``must`` and ``should`` clause lists are extended rather than replaced,
    so multiple tool calls accumulate their filters correctly.

    Args:
        state: The mutable AgentState dict.  ``state["dynamic_query"]`` is
            created if it does not yet exist.
        new_filter: A Qdrant filter dict with optional ``"must"`` and/or
            ``"should"`` keys, each containing a list of condition dicts.
            Empty dicts are silently ignored.
    """
    if not new_filter:
        return

    dynamic: Dict[str, Any] = state.setdefault("dynamic_query", {})
    for clause in ("must", "should"):
        if clause in new_filter:
            dynamic.setdefault(clause, []).extend(new_filter[clause])
