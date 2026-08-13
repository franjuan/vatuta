# Observability & Metrics

Vatuta uses **Prometheus** for tracking application metrics, particularly around data ingestion and API interactions.

This document describes the core metrics available for monitoring the health, performance,
and ingestion quality across all source integrations.

## Shared Source Metrics

All data sources (Slack, Jira, Confluence) report the following standard metrics for API and operation tracking:

### API and Network Operations

- **`source_api_calls_total`** (Counter)
  - **Description**: Total number of API calls made to the source.
  - **Labels**: `source`, `source_id`, `method`, `status`
- **`source_api_latency_seconds`** (Histogram)
  - **Description**: Latency of API calls to the source systems.
  - **Labels**: `source`, `source_id`, `method`, `status`

### Bulk Operations

- **`source_operation_latency_seconds`** (Histogram)
  - **Description**: Total duration of logical data collection operations.
  - **Labels**: `source`, `source_id`, `operation` (e.g., `collect_project`, `collect_space`, `collect_documents_and_chunks`)
- **`source_operation_items`** (Histogram)
  - **Description**: Number of items processed during a specific bulk operation.
  - **Labels**: `source`, `source_id`, `operation`

## Ingestion Quality Metrics

To monitor the RAG ingestion pipeline, prevent silent truncations, and ensure chunks are
optimally sized for embeddings, Vatuta tracks these metrics:

### Volume Metrics

- **`ingest_documents_total`** (Counter)
  - **Description**: Total number of high-level documents (pages, issues, threads) successfully ingested.
  - **Labels**: `source`, `source_id`
- **`ingest_chunks_total`** (Counter)
  - **Description**: Total number of individual chunks produced.
  - **Labels**: `source`, `source_id`, `chunk_type`
    - *Common chunk types*: `content` (Confluence), `slack_message` (Slack), `ticket`, `comment`,
      `history`, `relationship` (Jira)

### Sizing and Budget Metrics

- **`ingest_document_size_chars`** (Histogram)
  - **Description**: Character size distribution of incoming documents before chunking.
  - **Labels**: `source`, `source_id`
- **`ingest_chunk_size_chars`** (Histogram)
  - **Description**: Character size distribution of the produced chunks. Helps ensure chunk strategies are working.
  - **Labels**: `source`, `source_id`, `chunk_type`
- **`ingest_chunk_token_budget_ratio`** (Histogram)
  - **Description**: Ratio of chunk character length to the embedding model's maximum allowed context.
    Values `> 1.0` indicate high risk of silent truncation by the embedding model.
  - **Labels**: `source`, `source_id`, `chunk_type`
- **`ingest_chunks_per_document`** (Histogram)
  - **Description**: Number of chunks generated per individual document.
  - **Labels**: `source`, `source_id`

### Specific Chunking Metrics

- **`ingest_embedding_latency_seconds`** (Histogram)
  - **Description**: Latency for executing local embedding models during ingestion (e.g., semantic splitting).
  - **Labels**: `source`, `source_id`
- **`ingest_chunk_split_reason_total`** (Counter)
  - **Description**: Tracks the internal trigger that caused a chunk to be split, useful for tuning the chunking strategies.
  - **Labels**: `source`, `source_id`, `reason` (`time`, `size_chars`, `size_count`, `semantic`)

## Source-Specific Metrics

### Slack

- **`slack_user_cache_hits_total`** (Counter)
- **`slack_user_cache_misses_total`** (Counter)
  - Tracks the hit rate of the persistent user ID resolution cache.

## Usage & Best Practices

1. **Monitor `ingest_chunk_token_budget_ratio`**: This is your primary metric for ingestion health.
   If you see ratios crossing `1.0`, your `chunk_max_size_chars` config is too high for your
   current embedding model and you are losing data.
2. **Review `ingest_chunk_split_reason_total`**: If chunks are predominantly splitting on `size_chars`
   or `size_count`, you might need to adjust your thresholds to allow the `semantic` strategy
   to operate effectively.

## Logging Architecture

Vatuta uses Python's standard `logging` library configured via `logging.config.dictConfig` from YAML configuration files.

### Configuration (`config/logging.yaml`)

The default logging structure is defined in `config/logging.yaml`:

```yaml
version: 1
disable_existing_loggers: false

formatters:
  standard:
    format: "%(message)s"
    datefmt: "[%X]"

handlers:
  console:
    class: rich.logging.RichHandler
    level: DEBUG
    rich_tracebacks: true
    tracebacks_show_locals: false
    log_time_format: "[%X]"

root:
  level: INFO
  handlers:
    - console

loggers:
  src:
    level: INFO
  src.rag.agent:
    level: DEBUG
  qdrant_client:
    level: WARNING
  httpx:
    level: WARNING
  httpcore:
    level: WARNING
  docker:
    level: WARNING
  external.mcp:
    level: INFO
```

### Application Logger Namespaces

- **`src`**: Root namespace for all internal Vatuta modules (`src.rag`, `src.sources`, `src.client`, etc.).
- **`src.rag.agent`**: Detailed internal logging for RAG agent execution and decision paths.
- **`external.mcp.<server_name>`**: Loggers capturing stderr output from isolated MCP Docker container processes
  (e.g. `external.mcp.everything`, `external.mcp.wikipedia`).

### CLI Controls

- `--log-config <path>`: Specify a custom YAML logging configuration file.
- `-v` / `--verbose`: Force root and `src` loggers to `DEBUG` level during CLI invocation.

### Logging Formatting Standard

All internal modules use **lazy printf-style formatting** (`logger.info("Message %s", arg)`) rather than string
interpolation (`f"..."`) to ensure formatting operations are deferred until log evaluation.
