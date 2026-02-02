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
import re
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Final, List, Optional, Set, Tuple

from jira import JIRA
from pydantic import Field
from sentence_transformers.SentenceTransformer import SentenceTransformer
from sentence_transformers.util import cos_sim

from src.entities.manager import EntityManager

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

    # Comment chunking configuration
    chunk_max_size_chars: int = Field(default=2000, description="Max characters per comment chunk")
    chunk_max_count: int = Field(default=10, description="Max comments per comment chunk")
    chunk_similarity_threshold: float = Field(
        default=0.15, description="Cosine similarity threshold for comment chunk splitting"
    )
    chunk_embedding_model: str = Field(default="all-MiniLM-L6-v2", description="Model for semantic embeddings")


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
    def create(
        cls,
        config: JiraConfig,
        data_dir: str,
        secrets: Optional[dict] = None,
        entity_manager: Optional[EntityManager] = None,
    ) -> "JiraSource":
        """Create a JiraSource instance with the given configuration."""
        storage_path = os.path.join(data_dir, "jira")

        # Secrets handling
        if secrets is None:
            secrets = {}
        if "jira_user" not in secrets:
            secrets["jira_user"] = os.getenv("JIRA_USER")
        if "jira_api_token" not in secrets:
            secrets["jira_api_token"] = os.getenv("JIRA_API_TOKEN")

        return cls(config=config, secrets=secrets, storage_path=storage_path, entity_manager=entity_manager)

    def load_checkpoint(self) -> JiraCheckpoint:
        """Load existing checkpoint or create a new one."""
        from pathlib import Path

        cp_path = Path(self.storage_path) / self.source_id / "checkpoint.json"

        if cp_path.exists():
            checkpoint = JiraCheckpoint.load(cp_path, self.config)
        else:
            checkpoint = JiraCheckpoint(config=self.config)
        return checkpoint

    def __init__(
        self,
        config: JiraConfig,
        secrets: dict,
        storage_path: str | None = None,
        entity_manager: Optional[EntityManager] = None,
    ):
        """Initialize JIRA source with configuration and credentials."""
        if storage_path is None:
            raise ValueError("storage_path must be provided")
        super().__init__(config, secrets, storage_path=storage_path, entity_manager=entity_manager)

        # Config is already validated via type hint and generic

        user = self.secrets.get("jira_user") or os.getenv("JIRA_USER")
        token = self.secrets.get("jira_api_token") or os.getenv("JIRA_API_TOKEN")

        if not user or not token:
            raise ValueError("JIRA credentials (user/token) not found in secrets or environment")

        self.client = JIRA(
            server=self.config.url,
            basic_auth=(user, token),
            validate=False,
            get_server_info=False,
        )
        self._embedding_model: SentenceTransformer | None = None

        self._connection_validated = False

    def _resolve_user_entity(self, user_obj: Any) -> Optional[str]:
        """Resolve a JIRA user object to a global entity ID."""
        if not self.entity_manager or not user_obj:
            return None

        # user_obj might be a dict or a JIRA Resource object
        if hasattr(user_obj, "raw"):
            user_data = user_obj.raw
        elif isinstance(user_obj, dict):
            user_data = user_obj
        else:
            return None

        account_id = user_data.get("accountId") or user_data.get("key") or user_data.get("name")
        if not account_id:
            return None

        # Prepare user metadata
        name = user_data.get("displayName")
        email = user_data.get("emailAddress")

        # If email missing, try to fetch full profile if we have accountId (and it's not a key like 'admin')
        # Note: JIRA cloud uses accountId. Server uses name/key.
        if not email and "accountId" in user_data:
            try:
                # TODO: Try to fetch full user to get email if hidden in summary view
                # Only if we suspect we can get it.
                # This adds API calls. Let's be careful.
                # For now, rely on what we have.
                pass
            except Exception:
                pass

        data = {
            "name": name,
            "display_name": name,
            "email": email,
            "active": user_data.get("active"),
            "timeZone": user_data.get("timeZone"),
        }

        try:
            entity = self.entity_manager.get_or_create_user(
                source_type="jira", source_id=self.source_id, source_user_id=str(account_id), user_data=data
            )
            return entity.global_id
        except Exception as e:
            logger.warning(f"Failed to resolve entity for jira user {account_id}: {e}")
            return None

    def _ensure_connection(self) -> None:
        """Validate connection to JIRA if not already validated."""
        if self._connection_validated:
            return

        try:
            self.client.myself()
            self._connection_validated = True
        except Exception as e:
            raise ValueError(f"Failed to authenticate with JIRA: {e}") from e

    def get_specific_query(self, document_ids: List[str]) -> Optional[Dict[str, Any]]:
        """Check if any provided ID looks like a Jira key and return a filter."""
        if not document_ids:
            return None

        return {
            "must": [
                {"key": "metadata.source", "match": {"value": "jira"}},
                {"key": "metadata.source_doc_id", "match": {"any": document_ids}},
            ]
        }

    def collect_documents_and_chunks(
        self,
        checkpoint: Checkpoint,
        update_checkpoint: bool = True,
        use_cached_data: bool = True,
        filters: Optional[Dict[str, list[str]]] = None,
    ) -> Tuple[List[DocumentUnit], List[ChunkRecord]]:
        """Collect documents and chunks from JIRA projects."""
        self._ensure_connection()

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

    def _get_issue_fields_data(self, issue: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and normalize common fields from a Jira issue.

        Args:
            issue (Dict[str, Any]): The Jira issue dictionary.

        Returns:
            Dict[str, Any]: A dictionary containing the extracted and normalized fields.
        """
        fields = issue["fields"]

        def safe_get(obj: Any, key: str, default: str = "Unknown") -> str:
            if isinstance(obj, dict):
                return str(obj.get(key, default))
            return str(obj) if obj else default

        return {
            "key": issue["key"],
            "summary": fields.get("summary", ""),
            "status": safe_get(fields.get("status"), "name", "Unknown"),
            "priority": safe_get(fields.get("priority"), "name", "Unknown"),
            "issuetype": safe_get(fields.get("issuetype"), "name", "Unknown"),
            "assignee": safe_get(fields.get("assignee"), "displayName", "Unassigned"),
            "reporter": safe_get(fields.get("reporter"), "displayName", "Unknown"),
            "created": fields.get("created", ""),
            "updated": fields.get("updated", ""),
            "project_key": safe_get(fields.get("project"), "key", "Unknown"),
            "assignee_obj": fields.get("assignee"),
            "reporter_obj": fields.get("reporter"),
            "parent_obj": fields.get("parent"),
            "comment_obj": fields.get("comment"),
            "labels": fields.get("labels", []),
            "components": fields.get("components", []),
            "uri": issue.get("self", f"{self.config.url}/browse/{issue['key']}"),
        }

    def _build_system_tags(self, fields_data: Dict[str, Any], raw_fields: Dict[str, Any]) -> List[str]:
        """Build system tags including standard and configured fields.

        Args:
            fields_data (Dict[str, Any]): The fields data dictionary.
            raw_fields (Dict[str, Any]): The raw fields dictionary.

        Returns:
            List[str]: A list of system tags.
        """
        system_tags = [
            f"status:{fields_data['status']}",
            f"priority:{fields_data['priority']}",
            f"type:{fields_data['issuetype']}",
        ]

        # Standard derived tags to exclude from generic processing
        standard_derived_tags = {"status", "priority", "issuetype", "labels", "components"}

        for field_key in self.config.taggeable_fields:
            if field_key in standard_derived_tags:
                continue

            if field_key in raw_fields:
                val = raw_fields[field_key]
                if isinstance(val, list):
                    for v in val:
                        v_str = v.get("name", str(v)) if isinstance(v, dict) else str(v)
                        system_tags.append(f"{field_key}:{v_str}")
                elif isinstance(val, dict):
                    v_str = val.get("name", str(val))
                    system_tags.append(f"{field_key}:{v_str}")
                elif val is not None:
                    system_tags.append(f"{field_key}:{val}")

        # Labels
        for label in fields_data["labels"]:
            system_tags.append(f"label:{label}")

        # Components
        for comp in fields_data["components"]:
            comp_name = comp.get("name", "") if isinstance(comp, dict) else str(comp)
            if comp_name:
                system_tags.append(f"component:{comp_name}")

        return system_tags

    def _resolve_mentions(self, text: str) -> List[str]:
        """Extract user entities from mentions in text.

        Args:
            text (str): The text to extract mentions from.

        Returns:
            List[str]: A list of user entities.
        """
        if not self.entity_manager:
            return []

        mentions_re = re.compile(r"\[~(accountid:[a-zA-Z0-9:-]+|[^\]]+)\]")
        mentions = mentions_re.findall(text)

        user_tags = []
        for m in mentions:
            m_id = m.replace("accountid:", "")
            if len(m_id) > 5:
                try:
                    ent = self.entity_manager.get_user_by_source_id("jira", self.source_id, m_id)
                    if ent:
                        user_tags.append(f"user:{ent.global_id}")
                except Exception:
                    pass
        return user_tags

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
        data = self._get_issue_fields_data(issue)
        fields = issue["fields"]

        # 2. Build Base System Tags
        system_tags = self._build_system_tags(data, fields)

        # Parent ID logic
        parent_id = None
        if data["parent_obj"] and isinstance(data["parent_obj"], dict):
            parent_id = data["parent_obj"].get("key")

        # 3. Create Document Unit
        ticket_content = self._format_ticket_content(issue, names, schema)
        content_hash = DocumentUnit.compute_hash(ticket_content.encode("utf-8"))

        doc = DocumentUnit(
            document_id=f"jira|{self.source_id}|{data['key']}",
            source="jira",
            source_doc_id=data["key"],
            source_instance_id=self.source_id,
            uri=data["uri"],
            title=f"[{data['key']}] {data['summary']}",
            author=data["reporter"],
            parent_id=parent_id,
            language=None,
            source_created_at=self._parse_jira_time(data["created"]),
            source_updated_at=self._parse_jira_time(data["updated"]),
            system_tags=system_tags,
            source_metadata={
                "key": data["key"],
                "project": data["project_key"],
                "status": data["status"],
                "priority": data["priority"],
                "issuetype": data["issuetype"],
                "assignee": data["assignee"],
            },
            content_hash=content_hash,
        )

        # Tag doc with Reporter and Assignee Entities
        reporter_entity_id = self._resolve_user_entity(data["reporter_obj"])
        if reporter_entity_id:
            doc.system_tags.append(f"user:{reporter_entity_id}")

        assignee_entity_id = self._resolve_user_entity(data["assignee_obj"])
        if assignee_entity_id and assignee_entity_id != reporter_entity_id:
            doc.system_tags.append(f"user:{assignee_entity_id}")

        chunks: List[ChunkRecord] = []
        chunk_idx = 0

        # Chunk 1: Ticket Body
        chunk_tags = system_tags + ["type:ticket"]

        # Resolve mentions
        chunk_tags.extend(self._resolve_mentions(ticket_content))

        chunks.append(
            self._make_chunk(
                doc=doc,
                chunk_index=chunk_idx,
                text=ticket_content,
                tags=chunk_tags,
                created=doc.source_created_at,
                updated=doc.source_updated_at,
            )
        )
        chunk_idx += 1

        # Chunk 2: Relationships
        rel_text, rel_tags = self._format_relationships(issue)
        if rel_text:
            chunks.append(
                self._make_chunk(
                    doc=doc,
                    chunk_index=chunk_idx,
                    text=rel_text,
                    tags=system_tags + ["type:relationship"] + rel_tags,
                    created=doc.source_updated_at,  # Relationships change implies update
                    updated=doc.source_updated_at,
                )
            )
            chunk_idx += 1

        # Chunk 3+: History
        history_chunks_data = self._format_history(issue)
        for hist_text, hist_tags, hist_ts in history_chunks_data:
            chunks.append(
                self._make_chunk(
                    doc=doc,
                    chunk_index=chunk_idx,
                    text=hist_text,
                    tags=system_tags + ["type:history"] + hist_tags,
                    created=self._parse_jira_time(str(hist_ts)) if hist_ts else None,
                    updated=self._parse_jira_time(str(hist_ts)) if hist_ts else None,
                )
            )
            chunk_idx += 1

        # Chunk N+: Comments
        if self.config.include_comments and data["comment_obj"] and isinstance(data["comment_obj"], dict):
            comments = data["comment_obj"].get("comments", [])
            comment_chunks = self._build_chunks_for_comments(doc, comments, system_tags, start_index=chunk_idx)
            chunks.extend(comment_chunks)

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

    def _format_history(self, issue: Dict[str, Any]) -> List[Tuple[str, List[str], str]]:
        """Format changelog history into chunks.

        Returns:
            List of tuples: (text, tags, timestamp_string)
        """
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
            author_obj = history.get("author", {})
            author = author_obj.get("displayName", "Unknown") if isinstance(author_obj, dict) else "Unknown"
            created = history.get("created", "")

            # Resolve author entity
            author_entity_id = self._resolve_user_entity(author_obj)

            # This history entry's text lines
            entry_lines = []

            # Tags for this entry
            entry_tags = {f"author:{author}"}
            if author_entity_id:
                entry_tags.add(f"user:{author_entity_id}")

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
                # Store the timestamp of the first item in the batch for 'created' approximation
                first_ts_in_batch = history.get("created", "")
                chunks_data.append(("\n".join(batch_histories), list(batch_tags), first_ts_in_batch))
                batch_histories = []
                batch_tags = set()

        # Remaining
        if batch_histories:
            # We'll use the last known created (from loop) or fallback
            ts = histories[-1].get("created", "") if histories else ""
            chunks_data.append(("\n".join(batch_histories), list(batch_tags), ts))

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

    def _get_embedding_model(self) -> SentenceTransformer:
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(self.config.chunk_embedding_model)
        return self._embedding_model

    def _create_comment_chunk(
        self,
        doc: DocumentUnit,
        comments: List[dict],
        base_tags: List[str],
        chunk_index: int,
    ) -> ChunkRecord:
        """Create a chunk from a list of comments.

        Args:
            doc: The parent document.
            comments: List of comments in this chunk.
            base_tags: Base system tags to apply.
            chunk_index: The index for this chunk.

        Returns:
            ChunkRecord: The created chunk.
        """
        chunk_text = "\n\n".join([self._format_comment_for_embedding(cm)[0] for cm in comments])

        all_tags = set(base_tags)
        all_tags.add("type:comment")

        for cm in comments:
            # Comment content tags (author name)
            _, c_tags = self._format_comment_for_embedding(cm)
            all_tags.update(c_tags)

            # Author Entity
            author_obj = cm.get("author", {})
            author_ent_id = self._resolve_user_entity(author_obj)
            if author_ent_id:
                all_tags.add(f"user:{author_ent_id}")

            # Mentions
            all_tags.update(self._resolve_mentions(cm.get("body", "")))

        c_created_first = comments[0].get("created")
        c_created_last = comments[-1].get("created")

        return self._make_chunk(
            doc,
            chunk_index,
            chunk_text,
            list(all_tags),
            created=self._parse_jira_time(str(c_created_first)) if c_created_first else None,
            updated=self._parse_jira_time(str(c_created_last)) if c_created_last else None,
        )

    def _should_split_chunk(
        self,
        current_count: int,
        current_size: int,
        next_len: int,
        current_emb: Any,
        prev_emb: Optional[Any],
    ) -> bool:
        """Determine if a new chunk should be started.

        Args:
            current_count: Number of comments in the current chunk.
            current_size: Total character size of the current chunk.
            next_len: Length of the next comment to add.
            current_emb: Embedding of the next comment.
            prev_emb: Embedding of the previous comment (if any).

        Returns:
            True if the chunk should be split, False otherwise.
        """
        # 0. If current chunk is empty, can't split
        if current_count == 0:
            return False

        # 1. Size/Count-based splitting
        if (current_size + next_len > self.config.chunk_max_size_chars) or (
            current_count + 1 > self.config.chunk_max_count
        ):
            return True

        # 2. Semantic splitting
        if prev_emb is not None:
            similarity = cos_sim(current_emb, prev_emb).item()
            if similarity < self.config.chunk_similarity_threshold:
                return True

        return False

    def _build_chunks_for_comments(
        self, doc: DocumentUnit, comments: List[dict], base_tags: List[str], start_index: int = 0
    ) -> List[ChunkRecord]:
        if not comments:
            return []

        sorted_comments = sorted(comments, key=lambda c: str(c.get("created", "")))
        texts_for_embedding = [c.get("body", "") or " " for c in sorted_comments]
        model = self._get_embedding_model()
        embeddings = model.encode(texts_for_embedding)

        chunks: List[ChunkRecord] = []
        current_chunk_comments: List[dict] = []
        current_chunk_size = 0
        chunk_counter = start_index
        prev_comment_emb: Optional[Any] = None

        for _idx, (c, emb) in enumerate(zip(sorted_comments, embeddings, strict=False)):
            text, _ = self._format_comment_for_embedding(c)
            comment_len = len(text)

            if self._should_split_chunk(
                len(current_chunk_comments), current_chunk_size, comment_len, emb, prev_comment_emb
            ):
                chunks.append(self._create_comment_chunk(doc, current_chunk_comments, base_tags, chunk_counter))
                chunk_counter += 1

                current_chunk_comments = []
                current_chunk_size = 0

            current_chunk_comments.append(c)
            current_chunk_size += comment_len
            prev_comment_emb = emb

        if current_chunk_comments:
            chunks.append(self._create_comment_chunk(doc, current_chunk_comments, base_tags, chunk_counter))

        return chunks

    def _make_chunk(
        self,
        doc: DocumentUnit,
        chunk_index: int,
        text: str,
        tags: List[str],
        created: Optional[datetime] = None,
        updated: Optional[datetime] = None,
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
            source_created_at=created,
            source_updated_at=updated,
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
    jira_url = os.getenv("JIRA_INSTANCE_URL", "")
    jira_projects = os.getenv("JIRA_PROJECTS", "").split(",") if os.getenv("JIRA_PROJECTS") else []
    jira_user = os.getenv("JIRA_USER")
    jira_api_token = os.getenv("JIRA_API_TOKEN")

    storage_path = "./data/jira"
    source_id = "jira-main"

    if not jira_url or not jira_projects or not jira_user:
        print("Please set JIRA_INSTANCE_URL, JIRA_PROJECTS (comma-separated), JIRA_USER, and JIRA_API_TOKEN")
        return

    secrets = {"jira_user": jira_user, "jira_api_token": jira_api_token}

    config = JiraConfig(
        id=source_id,
        url=jira_url,
        projects=jira_projects,
        initial_lookback_days=365,
        include_comments=True,
        use_cached_data=False,
        taggeable_fields=["priority", "issuetype", "status", "assignee", "reporter", "labels"],
        chunk_max_size_chars=2000,
        chunk_max_count=10,
        chunk_similarity_threshold=0.15,
    )

    entity_manager = EntityManager(storage_path="data/entities.json")
    source = JiraSource(config, secrets, storage_path=storage_path, entity_manager=entity_manager)

    # Load checkpoint
    checkpoint_dir = Path(storage_path) / source_id
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
