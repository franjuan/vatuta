"""Prometheus metrics for monitoring source operations."""

import prometheus_client as _prom

Counter = _prom.Counter
Histogram = _prom.Histogram


# Unified API metrics (used by all sources)
API_LATENCY = Histogram(
    "source_api_latency_seconds",
    "Source API latency in seconds by source, source_id, method and status",
    ["source", "source_id", "method", "status"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, float("inf")),
)
API_CALLS = Counter(
    "source_api_calls_total",
    "Source API call count by source, source_id, method and status",
    ["source", "source_id", "method", "status"],
)

# Unified operation metrics (used by all sources)
OP_LATENCY = Histogram(
    "source_operation_latency_seconds",
    "Total latency of source operations by source, source_id and operation",
    ["source", "source_id", "operation"],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, float("inf")),
)
OP_ITEMS = Histogram(
    "source_operation_items",
    "Total number of items returned by source operations",
    ["source", "source_id", "operation"],
    buckets=(0, 1, 5, 10, 20, 50, 100, 200, 500, 1000, 5000, float("inf")),
)

# Slack-specific metrics (cache hits/misses)
USER_CACHE_HITS = Counter(
    "slack_user_cache_hits_total",
    "Slack user cache hits",
)
USER_CACHE_MISSES = Counter(
    "slack_user_cache_misses_total",
    "Slack user cache misses",
)

# Legacy metric names for backward compatibility (deprecated)
SLACK_API_LATENCY = API_LATENCY
SLACK_API_CALLS = API_CALLS
JIRA_API_LATENCY = API_LATENCY
JIRA_API_CALLS = API_CALLS
JIRA_OP_LATENCY = OP_LATENCY
JIRA_OP_ITEMS = OP_ITEMS
