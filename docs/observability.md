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
