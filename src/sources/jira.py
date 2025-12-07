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
from typing import Any, Dict, Final, List, Optional, Tuple

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

        next_page_token = None
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

        # 2. Build System Tags
        system_tags = []
        system_tags.append(f"status:{status}")
        system_tags.append(f"priority:{priority}")
        system_tags.append(f"type:{issuetype}")

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

        doc = DocumentUnit(
            document_id=f"jira|{self.source_id}|{key}",
            source="jira",
            source_doc_id=key,
            source_instance_id=self.source_id,
            uri=uri,
            title=f"[{key}] {summary}",
            author_name=reporter,
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
        )

        # 3. Create Chunks
        chunks: List[ChunkRecord] = []

        # Chunk 0: Description + Properties in markdown format
        primary_text = self._format_issue_for_embedding(issue)

        chunks.append(self._make_chunk(doc=doc, chunk_index=0, text=primary_text, tags=["type:description"]))

        # Chunk 1..N: Comments
        if self.config.include_comments:
            comment_obj = fields.get("comment")
            if comment_obj and isinstance(comment_obj, dict):
                comments = comment_obj.get("comments", [])
                for i, comment in enumerate(comments):
                    comment_text = self._format_comment_for_embedding(comment)
                    c_author_obj = comment.get("author", {})
                    c_author = (
                        c_author_obj.get("displayName", "Unknown") if isinstance(c_author_obj, dict) else "Unknown"
                    )

                    chunks.append(
                        self._make_chunk(
                            doc=doc, chunk_index=i + 1, text=comment_text, tags=["type:comment", f"author:{c_author}"]
                        )
                    )

        return doc, chunks

    def _format_properties(self, issue: Dict[str, Any]) -> str:
        """Format issue properties into a readable string."""
        fields = issue["fields"]
        lines = []

        issuetype = (
            fields.get("issuetype", {}).get("name", "Unknown")
            if isinstance(fields.get("issuetype"), dict)
            else str(fields.get("issuetype", "Unknown"))
        )
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

        assignee_obj = fields.get("assignee")
        assignee = assignee_obj.get("displayName", "Unassigned") if isinstance(assignee_obj, dict) else "Unassigned"

        reporter_obj = fields.get("reporter")
        reporter = reporter_obj.get("displayName", "Unknown") if isinstance(reporter_obj, dict) else "Unknown"

        lines.append(f"Type: {issuetype}")
        lines.append(f"Status: {status}")
        lines.append(f"Priority: {priority}")
        lines.append(f"Assignee: {assignee}")
        lines.append(f"Reporter: {reporter}")

        labels = fields.get("labels", [])
        if labels:
            lines.append(f"Labels: {', '.join(labels)}")

        components = fields.get("components", [])
        if components:
            comp_names = [c.get("name", "") if isinstance(c, dict) else str(c) for c in components]
            comp_names = [n for n in comp_names if n]
            if comp_names:
                lines.append(f"Components: {', '.join(comp_names)}")

        return "\n".join(lines)

    def _format_issue_for_embedding(self, issue: Dict[str, Any]) -> str:
        """Format issue as comprehensive markdown for embedding."""
        key = issue["key"]
        fields = issue["fields"]
        lines = []

        # Title
        summary = fields.get("summary", "")
        lines.append(f"# {key}: {summary}")
        lines.append("")

        # Metadata table
        lines.append("## Metadata")
        lines.append("")
        lines.append("| Property | Value |")
        lines.append("|----------|-------|")
        lines.append(f"| **Key** | {key} |")

        issuetype = (
            fields.get("issuetype", {}).get("name", "Unknown")
            if isinstance(fields.get("issuetype"), dict)
            else str(fields.get("issuetype", "Unknown"))
        )
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

        lines.append(f"| **Type** | {issuetype} |")
        lines.append(f"| **Status** | {status} |")
        lines.append(f"| **Priority** | {priority} |")

        project_obj = fields.get("project", {})
        project_key = project_obj.get("key", "Unknown") if isinstance(project_obj, dict) else "Unknown"
        lines.append(f"| **Project** | {project_key} |")

        assignee_obj = fields.get("assignee")
        assignee = assignee_obj.get("displayName", "Unassigned") if isinstance(assignee_obj, dict) else "Unassigned"
        lines.append(f"| **Assignee** | {assignee} |")

        reporter_obj = fields.get("reporter")
        reporter = reporter_obj.get("displayName", "Unknown") if isinstance(reporter_obj, dict) else "Unknown"
        lines.append(f"| **Reporter** | {reporter} |")

        created = fields.get("created", "")
        updated = fields.get("updated", "")
        lines.append(f"| **Created** | {created} |")
        lines.append(f"| **Updated** | {updated} |")

        # Labels
        labels = fields.get("labels", [])
        if labels:
            lines.append(f"| **Labels** | {', '.join(labels)} |")

        # Components
        components = fields.get("components", [])
        if components:
            comp_names = [c.get("name", "") if isinstance(c, dict) else str(c) for c in components]
            comp_names = [n for n in comp_names if n]
            if comp_names:
                lines.append(f"| **Components** | {', '.join(comp_names)} |")

        # Resolution
        resolution_obj = fields.get("resolution")
        if resolution_obj:
            resolution_name = (
                resolution_obj.get("name", str(resolution_obj))
                if isinstance(resolution_obj, dict)
                else str(resolution_obj)
            )
            lines.append(f"| **Resolution** | {resolution_name} |")

        # Fix versions
        fix_versions = fields.get("fixVersions", [])
        if fix_versions:
            version_names = [v.get("name", "") if isinstance(v, dict) else str(v) for v in fix_versions]
            version_names = [n for n in version_names if n]
            if version_names:
                lines.append(f"| **Fix Versions** | {', '.join(version_names)} |")

        # Affected versions
        versions = fields.get("versions", [])
        if versions:
            version_names = [v.get("name", "") if isinstance(v, dict) else str(v) for v in versions]
            version_names = [n for n in version_names if n]
            if version_names:
                lines.append(f"| **Affected Versions** | {', '.join(version_names)} |")

        lines.append("")

        # Description
        lines.append("## Description")
        lines.append("")
        description = fields.get("description", "*No description provided*")
        if not description:
            description = "*No description provided*"
        lines.append(description)
        return "\n".join(lines)

    def _format_comment_for_embedding(self, comment: Dict[str, Any]) -> str:
        """Format comment as markdown for embedding."""
        author_obj = comment.get("author", {})
        c_author = author_obj.get("displayName", "Unknown") if isinstance(author_obj, dict) else "Unknown"
        c_created = comment.get("created", "")
        c_body = comment.get("body", "")

        lines = []
        lines.append(f"### Comment by {c_author}")
        lines.append(f"*Posted on {c_created}*")
        lines.append("")
        lines.append(c_body)

        return "\n".join(lines)

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
        "projects": ["AL", "UA"],
        "storage_path": "./data/jira",
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
