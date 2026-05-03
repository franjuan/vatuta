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

# Ingestion quality metrics (shared across all sources)
INGEST_DOCUMENTS_TOTAL = Counter(
    "ingest_documents_total",
    "Total number of documents ingested per source and source_id",
    ["source", "source_id"],
)
INGEST_CHUNKS_TOTAL = Counter(
    "ingest_chunks_total",
    "Total number of chunks produced per source, source_id and chunk_type",
    ["source", "source_id", "chunk_type"],
)
INGEST_CHUNK_SIZE_CHARS = Histogram(
    "ingest_chunk_size_chars",
    "Distribution of chunk sizes in characters per source, source_id and chunk_type",
    ["source", "source_id", "chunk_type"],
    buckets=(50, 100, 200, 400, 600, 800, 1000, 1500, 2000, 3000, 5000, float("inf")),
)
INGEST_DOCUMENT_SIZE_CHARS = Histogram(
    "ingest_document_size_chars",
    "Distribution of document sizes in characters per source and source_id",
    ["source", "source_id"],
    buckets=(100, 300, 500, 1000, 2000, 4000, 8000, 16000, 32000, float("inf")),
)
INGEST_CHUNKS_PER_DOCUMENT = Histogram(
    "ingest_chunks_per_document",
    "Distribution of number of chunks per document per source and source_id",
    ["source", "source_id"],
    buckets=(1, 2, 3, 5, 7, 10, 15, 20, 30, 50, float("inf")),
)
INGEST_CHUNK_TOKEN_BUDGET_RATIO = Histogram(
    "ingest_chunk_token_budget_ratio",
    "Ratio of chunk chars to embedding model max chars (>1.0 risks truncation)",
    ["source", "source_id", "chunk_type"],
    buckets=(0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, float("inf")),
)
INGEST_EMBEDDING_LATENCY_SECONDS = Histogram(
    "ingest_embedding_latency_seconds",
    "Latency of embedding batch encoding during chunking per source and source_id",
    ["source", "source_id"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, float("inf")),
)
INGEST_CHUNK_SPLIT_REASON = Counter(
    "ingest_chunk_split_reason_total",
    "Count of chunk boundary triggers by reason (time, size_chars, size_count, semantic)",
    ["source", "source_id", "reason"],
)

# Legacy metric names for backward compatibility (deprecated)
SLACK_API_LATENCY = API_LATENCY
SLACK_API_CALLS = API_CALLS
JIRA_API_LATENCY = API_LATENCY
JIRA_API_CALLS = API_CALLS
JIRA_OP_LATENCY = OP_LATENCY
JIRA_OP_ITEMS = OP_ITEMS
