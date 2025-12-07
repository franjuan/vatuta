# ADR: Slack Source – Design and Decisions

Date: 2025-11-12
Status: Accepted

## Context

We ingest Slack conversations for RAG and analytics. The ingestion must be reliable under Slack rate limits, observable, and produce embeddings-friendly chunks while minimizing re-fetches and avoiding storing excessive PII or payload.

## Decision

1. Single-pass extraction (documents + chunks)
   - Implement a combined method `collect_documents_and_chunks(checkpoint, update_checkpoint=True, use_cached_data=True, filters=None) -> (List[DocumentUnit], List[ChunkRecord])` that fetches messages, builds logical documents, and immediately produces flat chunks (one per message).
   - Result: no separate chunking step and no re-fetch needed for chunking.

2. Document model
   - Documents represent either:
     - Threads (is_thread=True) identified by `thread_ts`
     - Channel messages (is_thread=False), grouped into N-minute windowed buckets using `channel_window_minutes`
   - Document permalink is a fallback URL derived from the first message timestamp:
     `https://{workspace_domain}/archives/{channel_id}/p{ts_seconds}{ts_micros_6}`
   - Messages are NOT stored in `DocumentUnit` metadata; only minimal attributes are kept.

3. Chunk model
   - Flat, single-level `ChunkRecord` per message (`level=0`, `parent_chunk_id=None`).
   - Chunk text format (LLM-friendly):
     `[YYYY-MM-DD HH:MM UTC] Display Name: normalized text`
   - Mentions `<@U...>` are resolved to `@DisplayName` using a local users cache.
   - Simple tags (URLs, email domains) are extracted via regex and set in `system_tags`.

4. Rate limiting
   - Use an adaptive, per-method rate limiter (token-bucket with jitter, 429-aware backoff, gradual recovery).
   - Methods tiering:
     - `conversations.list` → Tier 2 defaults
     - `conversations.history` → Tier 3 defaults
     - `users.info` → Tier 2 defaults
   - On HTTP 429, honor `Retry-After`, sleep, and notify limiter to back off.

5. Observability
   - Prometheus metrics:
     - `slack_api_latency_seconds{method,status}` (Histogram)
     - `slack_api_calls_total{method,status}` (Counter)
     - `slack_operation_latency_seconds{operation}` (Histogram)
     - `slack_operation_items{operation}` (Histogram)
     - `slack_user_cache_hits_total`, `slack_user_cache_misses_total` (Counters)
   - Structured INFO/DEBUG logs for pacing, cooldowns, and per-operation summaries.

6. Users cache (persistent)
   - Backed by a JSON persistent cache (`PersistentCache`) with TTL per entry.
   - Used for `users.info` display name resolution to avoid repeated API calls.

## Rationale

- Combining documents and chunk creation removes an extra pass and avoids re-fetching, improving latency and reducing rate-limit pressure.
- The adaptive limiter and Retry-After compliance maintain reliability under Slack’s per-method tiering.

## Consequences

- Pros:
  - Fewer API calls overall; faster end-to-end ingestion.
  - Better robustness to rate limits and network issues.
  - Cleaner, LLM-friendly chunk text with resolved mentions and lightweight tags.
  - Clear and minimal `DocumentUnit` footprint without messages in metadata.
- Cons:
  - Changing chunking strategy later requires re-running the combined extractor.
  - Fallback permalinks require `workspace_domain` configuration and may differ from Slack’s official `chat.getPermalink` in edge cases.

## Alternatives considered

- Two-step flow (collect-docs then chunk): increases complexity and risks drift; rejected.
- Storing messages in `DocumentUnit`: increases payload/PII and redundancy; rejected.
- Using Slack permalinks API for URIs: more rate limits and API calls; fallback URL is sufficient.

## Configuration

Required:

- `id` (string; unique identifier for this source instance)
- `workspace_domain` (e.g., `https://example.slack.com`)
- `channel_window_minutes` (int; window size for channel documents; default 60)
- `channel_types` (list[str]; e.g., `["public_channel","private_channel","im","mpim"]`)
- `user_cache_path` (string; JSON file path)
- `user_cache_ttl_seconds` (int; TTL for user entries; default 7 days)
- `initial_lookback_days` (int; days to look back when starting fresh; default 7)

Optional tier overrides:

- `enabled` (bool; default True)
- `tier2_rpm`, `tier2_cap` (affects `conversations.list`, `users.info`)
- `tier3_rpm`, `tier3_cap` (affects `conversations.history`)

## Interface summary

- `collect_documents_and_chunks(checkpoint, update_checkpoint=True, use_cached_data=True, filters=None) -> (documents, chunks)`
  - Documents: threads or channel windows
  - Chunks: one per message; flat, single-level
- `collect_cached_documents_and_chunks(filters=None, date_from=None, date_to=None)`
  - Replays only the locally cached JSONL files (no API calls, no checkpoint mutation).
  - Accepts optional channel filters and date bounds to regenerate documents/chunks from historical data.
- Private helpers:
  - Thread/window processors; user mention resolution; simple tag extraction; Slack permalink formatting; chunk builder.

## Raw data persistence

- Every ingestion stores the raw Slack payloads as gzipped JSON Lines under\
  `storage_path/<source_id>/<channel_id>/<YYYY>/<MM>/<channel_id>_<ts_start>_<ts_end>.jsonl.gz`.\
  This structure allows re-chunking or re-embedding without re-downloading from Slack.
- Cache-only rehydration uses these exact files, picking the ones whose timestamp window overlaps the user-specified range.

## Checkpointing

- `SlackCheckpoint` tracks `latest_ts` and `earliest_ts` **per channel**.
  - `latest_ts`: Used as the `oldest` parameter for `conversations.history` to fetch new messages. Defaults to `now - initial_lookback_days` if not present.
  - `earliest_ts`: Tracks the oldest message timestamp collected for the channel.
  After successfully writing the channel’s JSONL file, the checkpoint updates that specific channel and persists `checkpoint.json`.

## Open issues / future work

- Hierarchical chunking (e.g., group by day or by reply) using `parent_chunk_id` and `level>0`.
- Optional enrichment (NER/Pii) for tags at chunk or document level.
- Optional official permalinks via Slack API where needed.
