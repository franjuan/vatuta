"""JIRA source: issues and comments collection.

This module implements a JIRA data source that:
- Uses the `jira` library to fetch issues.
- Supports incremental collection via `updated` timestamp per project.
- Chunks issues into a primary chunk (description + properties) and comment chunks.
"""

import gzip
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Final, List, Optional, Set, Tuple

from jira import JIRA
from pydantic import Field

# Import Prometheus metrics from centralized module
from src.metrics.metrics import API_CALLS, API_LATENCY, OP_ITEMS, OP_LATENCY
from src.models.documents import ChunkRecord, DocumentUnit
from src.models.source_config import BaseSourceConfig
from src.sources.checkpoint import Checkpoint
from src.sources.source import Source

logger = logging.getLogger(__name__)


class JiraConfig(BaseSourceConfig):
    """Configuration for JIRA source."""

    url: str = Field(..., description="JIRA instance URL")
    jql_query: str = Field(
        default="project = '{project}' AND updated >= '{updated_since}' order by updated ASC",
        description="JQL query template. Use {project} and {updated_since} as placeholders.",
    )
    initial_lookback_days: int = Field(default=30, description="Initial lookback days for first run")
    include_comments: bool = Field(default=True, description="Whether to fetch and chunk comments")
    use_cached_data: bool = Field(default=False, description="Whether to persist/use cached data")
    projects: List[str] = Field(..., description="List of JIRA project keys to fetch")
    taggeable_fields: List[str] = Field(
        default_factory=list, description="List of issue fields to include as system tags"
    )

    history_chunk_size: int = Field(default=20, description="Max changelog items per history chunk")


class JiraCheckpoint(Checkpoint[JiraConfig]):
    """Checkpoint specifically for Jira source."""

    config: JiraConfig
    state: Dict[str, Any] = Field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize checkpoint state with default values."""
        self.state.setdefault("projects", {})
        default_updated_ts = self.__get_default_updated_ts()
        self.state.setdefault("default_updated_ts", float(default_updated_ts))

    def __get_default_updated_ts(self) -> float:
        return (datetime.now(timezone.utc) - timedelta(days=self.config.initial_lookback_days)).timestamp()

    def get_project_updated_ts(self, project_key: str) -> float:
        """Get the latest updated timestamp (seconds since epoch) for a project."""
        projects: Dict[str, Any] = self.state.setdefault("projects", {})
        value = projects.get(project_key)
        default_updated_ts = self.state.get("default_updated_ts", self.__get_default_updated_ts())
        if value is not None:
            return float(value)
        return float(default_updated_ts)

    def update_project_updated(self, project_key: str, ts: float) -> None:
        """Update the latest updated timestamp for a project if the new one is greater."""
        current = self.get_project_updated_ts(project_key)
        if ts > current:
            self.state["projects"][project_key] = ts


class JiraSource(Source[JiraConfig]):
    """JIRA source implementation."""

    JIRA_API_LIMIT: Final[int] = 50
    """Max issues per API call to JIRA"""

    @classmethod
    def create(cls, config: JiraConfig, data_dir: str, secrets: Optional[dict] = None) -> "JiraSource":
        """Create a JiraSource instance with the given configuration."""
        storage_path = os.path.join(data_dir, "jira")

        # Secrets handling
        if secrets is None:
            secrets = {}
        if "jira_user" not in secrets:
            secrets["jira_user"] = os.getenv("JIRA_USER")
        if "jira_api_token" not in secrets:
            secrets["jira_api_token"] = os.getenv("JIRA_API_TOKEN")

        return cls(config=config, secrets=secrets, storage_path=storage_path)

    def load_checkpoint(self) -> JiraCheckpoint:
        """Load existing checkpoint or create a new one."""
        from pathlib import Path

        cp_path = Path(self.storage_path) / self.source_id / "checkpoint.json"

        if cp_path.exists():
            checkpoint = JiraCheckpoint.load(cp_path, self.config)
        else:
            checkpoint = JiraCheckpoint(config=self.config)
        return checkpoint

    def __init__(self, config: JiraConfig, secrets: dict, storage_path: str | None = None):
        """Initialize JIRA source with configuration and credentials."""
        if storage_path is None:
            raise ValueError("storage_path must be provided")
        super().__init__(config, secrets, storage_path=storage_path)

        # Config is already validated via type hint and generic

        user = self.secrets.get("jira_user") or os.getenv("JIRA_USER")
        token = self.secrets.get("jira_api_token") or os.getenv("JIRA_API_TOKEN")

        if not user or not token:
            raise ValueError("JIRA credentials (user/token) not found in secrets or environment")

        self.client = JIRA(
            server=self.config.url,
            basic_auth=(user, token),
        )

        # Validate credentials immediately
        try:
            self.client.myself()
        except Exception as e:
            raise ValueError(f"Failed to authenticate with JIRA: {e}") from e

    def collect_documents_and_chunks(
        self,
        checkpoint: Checkpoint,
        update_checkpoint: bool = True,
        use_cached_data: bool = True,
        filters: Optional[Dict[str, list[str]]] = None,
    ) -> Tuple[List[DocumentUnit], List[ChunkRecord]]:
        """Collect documents and chunks from JIRA projects."""
        if not isinstance(checkpoint, JiraCheckpoint):
            # If passed a generic checkpoint, try to wrap it or start fresh if empty
            checkpoint = JiraCheckpoint(config=self.config, state=checkpoint.state if checkpoint else {})

        all_docs: List[DocumentUnit] = []
        all_chunks: List[ChunkRecord] = []

        for project in self.config.projects:
            if filters and "project_ids" in filters and project not in filters["project_ids"]:
                continue
            logger.info(f"Collecting documents for project: {project}")
            docs, chunks = self._collect_project(
                project, checkpoint, update_checkpoint=update_checkpoint, use_cached_data=use_cached_data
            )
            all_docs.extend(docs)
            all_chunks.extend(chunks)

        return all_docs, all_chunks

    def _collect_project(
        self, project: str, checkpoint: JiraCheckpoint, update_checkpoint: bool = True, use_cached_data: bool = True
    ) -> Tuple[List[DocumentUnit], List[ChunkRecord]]:
        op_start = perf_counter()
        latest_ts = checkpoint.get_project_updated_ts(project)

        # Build JQL using string formatting
        since_dt = datetime.fromtimestamp(latest_ts, tz=timezone.utc)
        updated_since = since_dt.strftime("%Y-%m-%d %H:%M")

        jql = self.config.jql_query.format(project=project, updated_since=updated_since)

        logger.debug(f"Searching JIRA with JQL: {jql}")

        # Paginate through all results using nextPageToken
        docs: List[DocumentUnit] = []
        chunks: List[ChunkRecord] = []
        max_updated_ts = latest_ts

        next_page_token: Optional[str] = None
        max_results = JiraSource.JIRA_API_LIMIT
        total_fetched = 0

        while True:
            # Track API call timing
            call_start = perf_counter()
            try:
                # Build parameters for enhanced_search_issues
                search_params = {
                    "jql_str": jql,
                    "maxResults": max_results,
                    "expand": "names,schema,changelog,transitions,properties",
                    "json_result": True,
                }
                if next_page_token:
                    search_params["nextPageToken"] = next_page_token

                result = self.client.enhanced_search_issues(**search_params)
                status = "200"
            except Exception as e:
                status = "error"
                elapsed = perf_counter() - call_start
                API_CALLS.labels(
                    source="jira", source_id=self.source_id, method="enhanced_search_issues", status=status
                ).inc()
                API_LATENCY.labels(
                    source="jira", source_id=self.source_id, method="enhanced_search_issues", status=status
                ).observe(elapsed)
                logger.error(f"JIRA API error: {e}")
                raise

            # Record successful API call metrics
            elapsed = perf_counter() - call_start
            API_CALLS.labels(
                source="jira", source_id=self.source_id, method="enhanced_search_issues", status=status
            ).inc()
            API_LATENCY.labels(
                source="jira", source_id=self.source_id, method="enhanced_search_issues", status=status
            ).observe(elapsed)

            # Extract issues from result
            issues = result["issues"]
            names = result["names"]
            schema = result["schema"]

            if not issues:
                break

            logger.debug(f"Fetched {len(issues)} issues (total: {total_fetched + len(issues)})")

            for issue in issues:
                doc, issue_chunks = self._process_issue(issue, names, schema)
                docs.append(doc)
                chunks.extend(issue_chunks)

                # Track latest updated timestamp
                updated_str = issue["fields"]["updated"]
                if updated_str:
                    try:
                        ts = datetime.strptime(updated_str, "%Y-%m-%dT%H:%M:%S.%f%z").timestamp()
                        if ts > max_updated_ts:
                            max_updated_ts = ts
                    except ValueError:
                        logger.warning(f"Could not parse updated timestamp: {updated_str}")

            total_fetched += len(issues)

            # Check if we've fetched all results (less than max means we're done)
            if result.get("nextPageToken", None) is None:
                break
            else:
                next_page_token = result["nextPageToken"]

        # Record operation metrics
        op_elapsed = perf_counter() - op_start
        logger.info(f"Completed collection for project {project}: {total_fetched} issues in {op_elapsed:.3f}s")
        OP_LATENCY.labels(source="jira", source_id=self.source_id, operation="collect_project").observe(op_elapsed)
        OP_ITEMS.labels(source="jira", source_id=self.source_id, operation="collect_project").observe(total_fetched)

        if use_cached_data:
            self.__persist_api_messages(
                project_key=project,
                new_issues=issues,
                base_dir=Path(self.storage_path) / self.source_id,
            )

        if update_checkpoint:
            checkpoint.update_project_updated(project, max_updated_ts)

        return docs, chunks

    def __persist_api_messages(
        self,
        project_key: str,
        new_issues: List[dict],
        base_dir: Path,
    ) -> None:
        """Persist fetched issues to a gzipped JSONL file, merging with existing data.

        Args:
            project_key: The JIRA project key.
            new_issues: List of new issue dictionaries.
            base_dir: Base directory for storage.
        """
        if not new_issues:
            return

        project_dir = base_dir / project_key
        project_dir.mkdir(parents=True, exist_ok=True)
        out_path = project_dir / f"{project_key}.jsonl.gz"

        # Load existing issues
        existing_issues = {}
        if out_path.exists():
            try:
                with gzip.open(out_path, "rt", encoding="utf-8") as f:
                    for line in f:
                        issue = json.loads(line)
                        existing_issues[issue["key"]] = issue
            except Exception as e:
                logger.error(f"Failed to load existing issues from {out_path}: {e}")

        # Update with new issues
        for issue in new_issues:
            existing_issues[issue["key"]] = issue

        # Write back to file
        try:
            with gzip.open(out_path, "wt", encoding="utf-8") as f:
                for issue in existing_issues.values():
                    f.write(json.dumps(issue, separators=(",", ":"), ensure_ascii=False) + "\n")
            logger.info(f"Persisted {len(existing_issues)} issues to {out_path}")
        except Exception as e:
            logger.error(f"Failed to persist issues to {out_path}: {e}")

    def _process_issue(
        self, issue: Dict[str, Any], names: Dict[str, str], schema: Dict[str, Any]
    ) -> Tuple[DocumentUnit, List[ChunkRecord]]:
        """Convert a JIRA issue into a DocumentUnit and ChunkRecords.

        Args:
            issue (Dict[str, Any]): The JIRA issue to process.
            names (Dict[str,str]): The names of the fields.
            schema (Dict[str, Any]): The schema of the issue.

        Returns:
            Tuple[DocumentUnit, List[ChunkRecord]]: The document unit and chunk records.
        """
        # 1. Extract Metadata
        key = issue["key"]
        fields = issue["fields"]

        summary = fields.get("summary", "")

        # Extract field values with safe navigation
        status = (
            fields.get("status", {}).get("name", "Unknown")
            if isinstance(fields.get("status"), dict)
            else str(fields.get("status", "Unknown"))
        )
        priority = (
            fields.get("priority", {}).get("name", "Unknown")
            if isinstance(fields.get("priority"), dict)
            else str(fields.get("priority", "Unknown"))
        )
        issuetype = (
            fields.get("issuetype", {}).get("name", "Unknown")
            if isinstance(fields.get("issuetype"), dict)
            else str(fields.get("issuetype", "Unknown"))
        )

        assignee_obj = fields.get("assignee")
        assignee = assignee_obj.get("displayName", "Unassigned") if isinstance(assignee_obj, dict) else "Unassigned"

        reporter_obj = fields.get("reporter")
        reporter = reporter_obj.get("displayName", "Unknown") if isinstance(reporter_obj, dict) else "Unknown"

        created = fields.get("created", "")
        updated = fields.get("updated", "")

        project_obj = fields.get("project", {})
        project_key = project_obj.get("key", "Unknown") if isinstance(project_obj, dict) else "Unknown"

        # 2. Build Base System Tags (Ticket Level)
        system_tags = []
        system_tags.append(f"status:{status}")
        system_tags.append(f"priority:{priority}")
        system_tags.append(f"type:{issuetype}")

        # Deduplication: standard fields that are already captured
        standard_derived_tags = {"status", "priority", "issuetype", "labels", "components"}

        # Add configured taggeable fields, carefully excluding duplicates if they are standard fields
        # Note: 'status', 'priority' etc are field keys. 'labels' is 'labels'.

        for field_key in self.config.taggeable_fields:
            if field_key in standard_derived_tags:
                continue

            if field_key in fields:
                val = fields[field_key]
                # Handle dicts (like components, versions) or simple values
                if isinstance(val, list):
                    for v in val:
                        v_str = v.get("name", str(v)) if isinstance(v, dict) else str(v)
                        system_tags.append(f"{field_key}:{v_str}")
                elif isinstance(val, dict):
                    v_str = val.get("name", str(val))
                    system_tags.append(f"{field_key}:{v_str}")
                elif val is not None:
                    system_tags.append(f"{field_key}:{val}")

        # Always include labels and components if present as they are standard
        labels = fields.get("labels", [])
        if labels:
            for label in labels:
                system_tags.append(f"label:{label}")

        components = fields.get("components", [])
        if components:
            for comp in components:
                comp_name = comp.get("name", "") if isinstance(comp, dict) else str(comp)
                if comp_name:
                    system_tags.append(f"component:{comp_name}")

        uri = issue.get("self", f"{self.config.url}/browse/{key}")

        # Parent ID logic
        parent_obj = fields.get("parent")
        parent_id = None
        if parent_obj and isinstance(parent_obj, dict):
            parent_id = parent_obj.get("key")

        # 3. Create Document Unit
        # We use the key-value representation for the primary hash
        ticket_content = self._format_ticket_content(issue, names, schema)
        content_hash = DocumentUnit.compute_hash(ticket_content.encode("utf-8"))

        doc = DocumentUnit(
            document_id=f"jira|{self.source_id}|{key}",
            source="jira",
            source_doc_id=key,
            source_instance_id=self.source_id,
            uri=uri,
            title=f"[{key}] {summary}",
            author=reporter,
            parent_id=parent_id,
            language=None,
            source_created_at=self._parse_jira_time(created),
            source_updated_at=self._parse_jira_time(updated),
            system_tags=system_tags,
            source_metadata={
                "key": key,
                "project": project_key,
                "status": status,
                "priority": priority,
                "issuetype": issuetype,
                "assignee": assignee,
            },
            content_hash=content_hash,
        )

        chunks: List[ChunkRecord] = []
        chunk_idx = 0

        # Chunk 1: Ticket Body (Key-Value)
        chunk_tags = system_tags + ["type:ticket"]
        chunks.append(self._make_chunk(doc=doc, chunk_index=chunk_idx, text=ticket_content, tags=chunk_tags))
        chunk_idx += 1

        # Chunk 2: Relationships
        rel_text, rel_tags = self._format_relationships(issue)
        if rel_text:
            chunks.append(
                self._make_chunk(
                    doc=doc, chunk_index=chunk_idx, text=rel_text, tags=system_tags + ["type:relationship"] + rel_tags
                )
            )
            chunk_idx += 1

        # Chunk 3+: History (Changelog) - Now split into multiple chunks if needed
        history_chunks_data = self._format_history(issue)
        for hist_text, hist_tags in history_chunks_data:
            chunks.append(
                self._make_chunk(
                    doc=doc, chunk_index=chunk_idx, text=hist_text, tags=system_tags + ["type:history"] + hist_tags
                )
            )
            chunk_idx += 1

        # Chunk N+: Comments
        if self.config.include_comments:
            comment_obj = fields.get("comment")
            if comment_obj and isinstance(comment_obj, dict):
                comments = comment_obj.get("comments", [])
                for comment in comments:
                    comment_text, comment_tags = self._format_comment_for_embedding(comment)
                    chunks.append(
                        self._make_chunk(
                            doc=doc,
                            chunk_index=chunk_idx,
                            text=comment_text,
                            tags=system_tags + ["type:comment"] + comment_tags,
                        )
                    )
                    chunk_idx += 1

        return doc, chunks

    def _format_ticket_content(self, issue: Dict[str, Any], names: Dict[str, str], schema: Dict[str, Any]) -> str:
        """Format ticket fields as Key: Value pairs.

        Includes all fields, preserving duplicates if necessary (though dict keys are unique),
        resolving names from schema/names, appending descriptions if available, and including empty values.
        """
        fields = issue["fields"]
        lines = []

        # Always start with Key for context
        lines.append(f"Key: {issue['key']}")

        # Proceed with all other fields
        # Note: fields dict keys are IDs (or keys) like 'summary', 'customfield_123', 'status'

        for field_id, val in fields.items():
            # Get human readable name
            field_name = names.get(field_id, field_id)

            # Check for description in schema
            # schema is typically {field_id: {type: ..., system: ..., items: ...}}
            # It usually doesn't have 'description'. But we'll check just in case or if user implies something else.
            # If unavailable, we just use the name.
            field_schema = schema.get(field_id, {})
            # Some JIRA instances might inject description here, or it might be in 'custom' or 'system' logic.
            # We'll assume if it's there it's under 'description'.
            field_desc = field_schema.get("description", "")

            label = field_name
            if field_desc:
                label = f"{field_name} ({field_desc})"

            # Format value
            str_val = ""
            if val is None:
                str_val = "None"  # Or empty string? "None" is explicit for "null".
            elif isinstance(val, dict):
                str_val = val.get("name") or val.get("key") or val.get("displayName") or val.get("value") or str(val)
            elif isinstance(val, list):
                # List of dicts or strings
                items = []
                for v in val:
                    if isinstance(v, dict):
                        items.append(v.get("name") or v.get("value") or str(v))
                    else:
                        items.append(str(v))
                str_val = ", ".join(items)
            else:
                str_val = str(val)

            # Sanitize str_val to prevent line breaks in the output
            str_val = str_val.replace("\r", " ").replace("\n", " ").strip()

            lines.append(f"{label}: {str_val}")

        return "\n".join(lines)

    def _format_relationships(self, issue: Dict[str, Any]) -> Tuple[str, List[str]]:
        """Extract relationships and return text content and tags."""
        fields = issue["fields"]
        lines = []
        tags = []

        # Parent
        parent = fields.get("parent")
        if parent:
            key = parent.get("key")
            lines.append(f"Parent: {key}")
            tags.append(f"rel:parent:{key}")

        # Subtasks
        subtasks = fields.get("subtasks", [])
        for task in subtasks:
            key = task.get("key")
            lines.append(f"Subtask: {key}")
            tags.append(f"rel:subtask:{key}")

        # Issue Links
        issuelinks = fields.get("issuelinks", [])
        for link in issuelinks:
            # Outward
            if "outwardIssue" in link:
                key = link["outwardIssue"]["key"]
                rel_type = link["type"]["outward"]
                lines.append(f"{rel_type}: {key}")
                tags.append(f"rel:{rel_type.replace(' ', '_').lower()}:{key}")
            # Inward
            if "inwardIssue" in link:
                key = link["inwardIssue"]["key"]
                rel_type = link["type"]["inward"]
                lines.append(f"{rel_type}: {key}")
                tags.append(f"rel:{rel_type.replace(' ', '_').lower()}:{key}")

        return "\n".join(lines), tags

    def _format_history(self, issue: Dict[str, Any]) -> List[Tuple[str, List[str]]]:
        """Format changelog history into chunks."""
        changelog = issue.get("changelog", {})
        if not changelog:
            return []

        histories = changelog.get("histories", [])
        if not histories:
            return []

        chunks_data = []

        # Current batch
        batch_histories: List[str] = []
        batch_tags: Set[str] = set()

        batch_size = self.config.history_chunk_size

        for history in histories:
            author = history.get("author", {}).get("displayName", "Unknown")
            created = history.get("created", "")

            # This history entry's text lines
            entry_lines = []

            # Tags for this entry
            entry_tags = {f"author:{author}"}

            items = history.get("items", [])
            if not items:
                continue

            for item in items:
                field = item.get("field", "")
                from_str = item.get("fromString", "")
                to_str = item.get("toString", "")

                entry_lines.append(f"{created} - {author} changed {field} from '{from_str}' to '{to_str}'")

                if field.lower() == "status":
                    entry_tags.add(f"transition:selection:{to_str}")

            # Add entry to batch
            # Join the items of this history entry with newlines
            batch_histories.append("\n".join(entry_lines))
            batch_tags.update(entry_tags)

            # Chunking by HISTORY entry count
            if len(batch_histories) >= batch_size:
                chunks_data.append(("\n".join(batch_histories), list(batch_tags)))
                batch_histories = []
                batch_tags = set()

        # Remaining
        if batch_histories:
            chunks_data.append(("\n".join(batch_histories), list(batch_tags)))

        return chunks_data

    def _format_comment_for_embedding(self, comment: Dict[str, Any]) -> Tuple[str, List[str]]:
        """Format comment as markdown for embedding and return tags."""
        author_obj = comment.get("author", {})
        c_author = author_obj.get("displayName", "Unknown") if isinstance(author_obj, dict) else "Unknown"
        c_created = comment.get("created", "")
        c_body = comment.get("body", "")

        lines = []
        lines.append(f"Author: {c_author}")
        lines.append(f"Created: {c_created}")
        lines.append("Content:")
        lines.append(c_body)

        tags = [f"author:{c_author}"]

        return "\n".join(lines), tags

    def _make_chunk(
        self,
        doc: DocumentUnit,
        chunk_index: int,
        text: str,
        tags: List[str],
    ) -> ChunkRecord:
        content_hash = DocumentUnit.compute_hash(text.encode("utf-8"))
        chunk_id = sha256(f"{doc.document_id}|{chunk_index}|{content_hash}".encode()).hexdigest()

        return ChunkRecord(
            chunk_id=chunk_id,
            parent_document_id=doc.document_id,
            chunk_index=chunk_index,
            text=text,
            system_tags=tags,
            content_hash=content_hash,
            level=0,
        )

    def _parse_jira_time(self, time_str: str) -> Optional[datetime]:
        if not time_str:
            return None
        try:
            return datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S.%f%z")
        except ValueError:
            return None

    def collect_cached_documents_and_chunks(
        self,
        filters: Optional[Dict[str, list[str]]] = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> Tuple[List[DocumentUnit], List[ChunkRecord]]:
        """Rehydrate documents and chunks exclusively from cached JSONL data.

        Args:
            filters: Optional list of identifiers by filtering type to restrict processing.
            date_from: Optional UTC datetime lower bound (inclusive) for issue update time.
            date_to: Optional UTC datetime upper bound (inclusive) for issue update time.

        Returns:
            Tuple[List[DocumentUnit], List[ChunkRecord]]: The documents discovered and
            their corresponding flat chunks.
        """
        base_dir = Path(self.storage_path) / self.source_id
        if not base_dir.exists():
            logger.warning(f"Cache directory {base_dir} does not exist; returning empty result")
            return [], []

        all_docs: List[DocumentUnit] = []
        all_chunks: List[ChunkRecord] = []

        # In Jira source, project_ids are treated as project keys
        if filters and "project_ids" in filters:
            target_projects = sorted(set(filters["project_ids"]))
        else:
            target_projects = [entry.name for entry in base_dir.iterdir() if entry.is_dir()]

        logger.info(
            f"Collecting cached documents for projects="
            f"{target_projects if filters and 'project_ids' in filters else 'ALL'} "
            f"range=({date_from}, {date_to})"
        )

        for project in target_projects:
            project_dir = base_dir / project
            file_path = project_dir / f"{project}.jsonl.gz"

            if not file_path.exists():
                logger.debug(f"No cache file found for project {project}")
                continue

            try:
                with gzip.open(file_path, "rt", encoding="utf-8") as f:
                    for line in f:
                        try:
                            issue = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        # Date filtering
                        if date_from or date_to:
                            updated_str = issue["fields"].get("updated")
                            created_str = issue["fields"].get("created")

                            updated_dt = self._parse_jira_time(updated_str)
                            created_dt = self._parse_jira_time(created_str)

                            # If we can't parse timestamps, we might default to including or excluding.
                            # Here we'll assume if we can't determine the time, we skip if strict,
                            # but let's try to be safe. If updated is missing, we can't filter by it.

                            if updated_dt and date_from and updated_dt < date_from:
                                # Issue ended before the window started -> No overlap
                                continue

                            if created_dt and date_to and created_dt > date_to:
                                # Issue started after the window ended -> No overlap
                                continue

                            # Note: If created_dt is None but updated_dt is present,
                            # we assume created <= updated.
                            # If updated_dt is None, we can't strictly check the "end" of the issue.
                            # But usually Jira issues have both.

                        # Process issue
                        # We pass empty dicts for names and schema as they are not used in _process_issue
                        # and not persisted in the cache file currently.
                        doc, chunks = self._process_issue(issue, {}, {})
                        all_docs.append(doc)
                        all_chunks.extend(chunks)

            except Exception as e:
                logger.error(f"Failed to read cache file {file_path}: {e}")

        logger.info(f"Cache-only replay completed: documents={len(all_docs)} chunks={len(all_chunks)}")
        return all_docs, all_chunks


def main() -> None:
    """Execute JIRA document collection."""
    logging.basicConfig(level=logging.INFO)

    # Example configuration from environment
    config = {
        "url": os.getenv("JIRA_INSTANCE_URL", ""),
        "projects": os.getenv("JIRA_PROJECTS", "").split(",") if os.getenv("JIRA_PROJECTS") else [],
        "storage_path": "./data/jira",
        "include_comments": True,
        "use_cached_data": False,
        "taggable_fields": ["priority", "issuetype", "status", "assignee", "reporter", "labels"],
        "initial_loopback_days": 365,
        "id": "jira-main",
    }

    secrets = {"jira_user": os.getenv("JIRA_USER"), "jira_api_token": os.getenv("JIRA_API_TOKEN")}

    if not config["url"] or not config["projects"] or not secrets["jira_user"]:
        print("Please set JIRA_INSTANCE_URL, JIRA_PROJECTS (comma-separated), JIRA_USER, and JIRA_API_TOKEN")
        return

    source = JiraSource(JiraConfig(**config), secrets, storage_path=str(config["storage_path"]))

    # Load checkpoint
    checkpoint_dir = Path(str(config["storage_path"])) / str(config["id"])
    checkpoint_path = checkpoint_dir / "checkpoint.json"
    checkpoint_data = {}
    if checkpoint_path.exists():
        with open(checkpoint_path, "r") as f:
            checkpoint_data = json.load(f)

    checkpoint = JiraCheckpoint(config=source.config, state=checkpoint_data)

    print("Starting collection...")
    docs, chunks = source.collect_documents_and_chunks(checkpoint)
    print(f"Collected {len(docs)} documents and {len(chunks)} chunks.")

    # Save checkpoint
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint.state, f, indent=2)
    print(f"Checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
