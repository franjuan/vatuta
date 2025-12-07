# ADR: Jira Source – Design and Decisions

Date: 2025-12-06
Status: Accepted

## Context

We ingest Jira issues to provide context on project tracking, bugs, and feature requests. Key requirements include capturing rich metadata (status, priority, assignee) and the conversational context within comments.

## Decision

1. **Single-pass extraction**
   - Fetches issues using JQL (Jira Query Language) with pagination.
   - Processes issues immediately into Documents and Chunks.

2. **Document model**
   - Each Jira Issue maps to one `DocumentUnit`.
   - `uri` links to the issue browse URL.
   - Metadata includes `key`, `project`, `status`, `priority`, `issuetype`, and `assignee`.
   - System tags allow filtering by `status`, `priority`, `type`, `label`, and `component`.

3. **Chunk model**
   - **Chunk 0 (Description)**: A formatted Markdown representation of the issue, including a metadata table (Key, Type, Status, etc.) and the full Description.
   - **Chunk 1..N (Comments)**: Each comment is a separate chunk, tagged with `type:comment` and `author:<name>`.

4. **Incremental Collection**
   - Uses `updated` timestamp via JQL parameters.
   - Checkpoints track the max `updated` timestamp per **Project**.

5. **Observability**
   - Prometheus metrics:
     - `jira_enhanced_search_issues_latency_seconds`, `jira_api_calls_total`
     - `jira_collect_project_latency_seconds`, `jira_collect_project_items`

## Rationale

- **Rich Metadata Embedding**: Including a metadata table in the primary chunk helps the LLM understand the state of the work item immediately.
- **Comments as Chunks**: Discussions in Jira comments often contain critical decisions or debugging details. treating them as separate chunks allows specific retrieval of these insights.
- **JQL Efficiency**: JQL is powerful for incremental fetching (`updated >= '...'`).

## Configuration

**Required:**

- `id` (str): Unique identifier for this source instance.
- `url` (str): Base URL of the Jira instance.
- `projects` (List[str]): List of Project keys to fetch.
- `secrets.jira_user` & `secrets.jira_api_token`: Authentication credentials.

**Optional:**

- `enabled` (bool): Default `True`.
- `jql_query` (str): Custom template. Defaults to `project = '{project}' AND updated >= '{updated_since}' order by updated ASC`.
- `initial_lookback_days` (int): Days, default 30.
- `include_comments` (bool): Default `True`.
- `use_cached_data` (bool): Default `False`.

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
