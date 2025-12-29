# ADR: Jira Source – Design and Decisions

Date: 2025-12-06
Status: Accepted

## Context

We ingest Jira issues to provide context on project tracking, bugs, and feature requests. Key
requirements include capturing rich metadata (status, priority, assignee) and the
conversational context within comments.

## Decision

- Fetches issues using JQL (Jira Query Language) with pagination.
- **Deferred Validation**: Connection and credentials are validated only when `collect_documents_and_chunks` is called,
  not on instantiation. This enables offline instantiation for cache-only workflows.
- Processes issues immediately into Documents and Chunks.

1. **Document model**
    - Each Jira Issue maps to one `DocumentUnit`.
    - `uri` links to the issue browse URL.
    - Metadata includes `key`, `project`, `status`, `priority`, `issuetype`, and `assignee`.
    - System tags automatically capture `status`, `priority`, `type`.
    - Configurable `taggeable_fields` allow specific custom fields to be promoted as `system_tags`.
    - `label` and `component` tags are added if present.

2. **Chunk model** (4 distinct chunk types)
    - **Chunk 0 (Ticket Body)**: A formatted Key-Value representation of the ticket content.
      - Includes ALL fields present in the issue.
      - Field names are mapped from the Jira schema to be human-readable.
      - Includes descriptions in parenthesis if available in schema.
      - Includes null/empty fields as "None".
      - Tags: `type:ticket` + metadata tags.
    - **Relationship Chunk**:
      - Lists parent, subtasks, and issue links (blocks, relates to, etc.).
      - Tags: `type:relationship` + relationship tags (e.g., `rel:blocks:KEY`, `rel:parent:KEY`).
    - **History Chunk(s)**:
      - Captures changelog history (author, field changes, timestamps).
      - Chunked by history entry count (configurable `history_chunk_size`, default 20 entries per chunk).
      - Tags: `type:history`, `author:<name>`, `transition:selection:<value>`.
    - **Comment Chunk(s)**:
      - Each comment is a separate chunk by default, but can be grouped/split based on strategy.
      - **Strategies**:
        - **Size/Count**: Merges comments into a single chunk until `chunk_max_size_chars` or `chunk_max_count` is reached.
        - **Semantic**: Splits chunks if the cosine similarity between consecutive comments drops below `chunk_similarity_threshold`.
      - Tags: `type:comment`, `author:<name>`, `author_id:<accountId>`.

3. **Incremental Collection**
    - Uses `updated` timestamp via JQL parameters.
    - Checkpoints track the max `updated` timestamp per **Project**.

4. **Observability**
    - Prometheus metrics:
      - `jira_enhanced_search_issues_latency_seconds`, `jira_api_calls_total`
      - `jira_collect_project_latency_seconds`, `jira_collect_project_items`

## Rationale

- **Granular Retrieval**: Splitting ticket content, relationships, history, and comments allows retrieval of specific aspects
(e.g., retrieving only history chunks when asking about "status changes" or "who worked on this").
- **Rich Metadata**: System tags enable powerful filtering on custom fields, relationships, and history authors.
- **Full Content Fidelity**: The ticket body chunk captures all fields, ensuring no custom data is lost,
using readable schema names for clarity.

## Configuration

**Required:**

- `id` (str): Unique identifier for this source instance.
- `url` (str): Base URL of the Jira instance.
- `projects` (List[str]): List of Project keys to fetch.
- `secrets.jira_user` & `secrets.jira_api_token`: Authentication credentials.

**Optional:**

- `enabled` (bool): Default `True`.
- `jql_query` (str): Custom template. Defaults to
`project = '{project}' AND updated >= '{updated_since}' order by updated ASC`.
- `initial_lookback_days` (int): Days, default 30.
- `include_comments` (bool): Default `True`.
- `use_cached_data` (bool): Default `False`.
- `taggeable_fields` (List[str]): List of issue field keys (e.g., `customfield_10001`, `priority`) to convert into `system_tags`.
Note: Standard fields like `status` are deduplicated.
- `history_chunk_size` (int): Max number of history entries per history chunk. Default 20.
- `chunk_max_size_chars` (int): Max characters per comment chunk. Default 2000.
- `chunk_max_count` (int): Max comments per comment chunk. Default 10.
- `chunk_similarity_threshold` (float): Cosine similarity threshold for splitting comment chunks. Default 0.15.
- `chunk_embedding_model` (str): SentenceTransformer model for embeddings. Default "all-MiniLM-L6-v2".

## Interface summary

- `collect_documents_and_chunks(checkpoint, update_checkpoint=True, use_cached_data=True, filters=None) -> (documents, chunks)`
  - **filters**: Supports `project_ids` (list of strings, mapped to project keys).
- `collect_cached_documents_and_chunks(filters=None, date_from=None, date_to=None)`
  - Replays locally cached JSONL files.

## Raw data persistence

- Issues are stored in gzipped JSONL files per project:
  `storage_path/<source_id>/<project_key>/<project_key>.jsonl.gz`
- New fetches append/merge into this single file per project (loading existing, updating, traversing).

## Checkpointing

- `JiraCheckpoint` stores a `projects` dictionary: `{ "PROJECT_KEY": <timestamp_float> }`.
- On each run, formats the timestamp for JQL.
- Updates the timestamp to the maximum `fields.updated` seen.

## Open issues / future work

- **Attachment Indexing**: Similar to Confluence, fetching text attachments.
- **Linked Issues**: explicit graph links in the document model for "blocks", "relates to".
