# ADR: Confluence Source – Design and Decisions

Date: 2025-12-06
Status: Accepted

## Context

We ingest Confluence pages for RAG and analytics. The ingestion must be reliable, handle incremental updates efficiently, and produce LLM-friendly markdown content from Confluence's internal HTML storage format.

## Decision

1. **Single-pass extraction**
   - `collect_documents_and_chunks` fetches pages via CQL (Confluence Query Language).
   - Pages are immediately converted to `DocumentUnit` and `ChunkRecord` objects.

2. **Document model**
   - Each Confluence Page maps to one `DocumentUnit`.
   - `uri` links to the web UI of the page.
   - Metadata includes `page_id`, `space`, and `title`.

3. **Chunk model**
   - **Content Chunk**: The main body of the page is converted from HTML to Markdown using `markdownify`.
   - Chunk text includes the page title as an H1 header.
   - Currently produces a single primary chunk for the body (future work may split by headers).

4. **Incremental Collection**
   - Uses `lastModified` timestamp via CQL.
   - Checkpoints track the max `lastModified` timestamp per **Space**.

5. **Observability**
   - Prometheus metrics:
     - `confluence_cql_latency_seconds`, `confluence_cql_calls_total`
     - `confluence_collect_space_latency_seconds`, `confluence_collect_space_items`

## Rationale

- **Markdown Conversion**: Confluence stores content as XHTML. Converting to Markdown is standard for RAG applications to reduce token usage and improve readability for LLMs.
- **Per-Space Checkpointing**: Spaces are the natural boundary for content in Confluence. Tracking updates per space allows for granular incremental syncs.

## Configuration

**Required:**

- `id` (str): Unique identifier for this source instance.
- `url` (str): Base URL of the Confluence instance.
- `spaces` (List[str]): List of Space keys to ingest.
- `secrets.jira_user` & `secrets.jira_api_token`: Authentication credentials (shared with Jira usually, or specific to Confluence).

**Optional:**

- `enabled` (bool): Default `True`.
- `use_cached_data` (bool): Default `False`.
- `initial_lookback_days` (int): Days to look back on the first run. If not specified (None), fetches **ALL** history.

## Interface summary

- `collect_documents_and_chunks(checkpoint, update_checkpoint=True, use_cached_data=True, filters=None) -> (documents, chunks)`
  - **filters**: Supports `space_ids` (list of strings) to restrict collection to specific spaces.
- `collect_cached_documents_and_chunks(filters=None, date_from=None, date_to=None)`
  - Replays locally cached page JSON files.

## Raw data persistence

- Pages are stored as individual gzipped JSON files:
  `storage_path/<source_id>/<space_key>/<page_id>.json.gz`
- This granular storage allows specific page re-processing but may result in many small files.

## Checkpointing

- `ConfluenceCheckpoint` stores a `spaces` dictionary: `{ "SPACE_KEY": <timestamp_float> }`.
- On each run, queries `lastModified >= <timestamp>`.
- Updates the timestamp to the maximum `version.when` seen in the batch.

## Open issues / future work

- **Hierarchical Chunking**: Split long pages by headers (H1, H2) into multiple chunks.
- **Comments**: Ingest page comments as separate chunks (similar to Jira).
- **Attachments**: Index text from PDF/Word attachments.
