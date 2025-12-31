"""Confluence source: pages and content collection.

This module implements a Confluence data source that:
- Uses the `atlassian-python-api` library to fetch pages.
- Supports incremental collection via `lastModified` timestamp per space.
- Converts HTML content to Markdown using regex (since no external lib available).
"""

import gzip
import json
import logging
import os
import re
from dataclasses import field
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Final, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from atlassian import Confluence
from markdownify import markdownify as md
from pydantic import Field

from src.entities.manager import EntityManager

# Import Prometheus metrics from centralized module
from src.metrics.metrics import API_CALLS, API_LATENCY, OP_ITEMS, OP_LATENCY
from src.models.documents import ChunkRecord, DocumentUnit
from src.models.source_config import BaseSourceConfig
from src.sources.checkpoint import Checkpoint
from src.sources.source import Source

logger = logging.getLogger(__name__)


class ConfluenceConfig(BaseSourceConfig):
    """Configuration for Confluence source."""

    url: str = Field(..., description="Confluence instance URL")
    spaces: List[str] = Field(..., description="List of Confluence space keys to fetch")
    use_cached_data: bool = Field(default=False, description="Whether to persist/use cached data")
    initial_lookback_days: Optional[int] = Field(
        default=None, description="Initial lookback days for first run. If None, fetch all."
    )
    chunk_max_size_chars: int = Field(default=1000, description="Max characters per chunk")
    chunk_similarity_threshold: float = Field(default=0.15, description="Cosine similarity threshold")
    chunk_overlap: int = Field(
        default=0, description="Overlap between chunks in characters (not used for header split)"
    )


class ConfluenceCheckpoint(Checkpoint[ConfluenceConfig]):
    """Checkpoint specifically for Confluence source."""

    config: ConfluenceConfig
    state: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize checkpoint state with default values."""
        self.state.setdefault("spaces", {})
        default_updated_ts = self.__get_default_updated_ts()
        self.state.setdefault("default_oldest_ts", float(default_updated_ts))

    def __get_default_updated_ts(self) -> float:
        return (datetime.now(timezone.utc) - timedelta(days=self.config.initial_lookback_days or 30)).timestamp()

    def get_space_updated_ts(self, space_key: str) -> float:
        """Get the latest updated timestamp (seconds since epoch) for a space."""
        spaces: Dict[str, Any] = self.state.setdefault("spaces", {})
        value = spaces.get(space_key)
        default_updated_ts = self.state.get("default_oldest_ts", self.__get_default_updated_ts())
        if value is not None:
            return float(value)
        return float(default_updated_ts)

    def update_space_updated(self, space_key: str, ts: float) -> None:
        """Update the latest updated timestamp for a space if the new one is greater."""
        current = self.get_space_updated_ts(space_key)
        if ts > current:
            self.state["spaces"][space_key] = ts


class ConfluenceSource(Source[ConfluenceConfig]):
    """Confluence source implementation."""

    CONFLUENCE_API_LIMIT: Final[int] = 10
    """Max pages per API call to Confluence"""

    @classmethod
    def create(
        cls,
        config: ConfluenceConfig,
        data_dir: str,
        secrets: Optional[dict] = None,
        entity_manager: Optional[EntityManager] = None,
    ) -> "ConfluenceSource":
        """Create a ConfluenceSource instance with the given configuration."""
        storage_path = os.path.join(data_dir, "confluence")

        # Secrets handling
        if secrets is None:
            secrets = {}
        if "jira_user" not in secrets:
            secrets["jira_user"] = os.getenv("JIRA_USER")
        if "jira_api_token" not in secrets:
            secrets["jira_api_token"] = os.getenv("JIRA_API_TOKEN")

        return cls(config=config, secrets=secrets, storage_path=storage_path, entity_manager=entity_manager)

    def load_checkpoint(self) -> ConfluenceCheckpoint:
        """Load existing checkpoint or create a new one."""
        from pathlib import Path

        cp_path = Path(self.storage_path) / self.source_id / "checkpoint.json"
        if cp_path.exists():
            confluence_checkpoint = ConfluenceCheckpoint.load(cp_path, self.config)
        else:
            confluence_checkpoint = ConfluenceCheckpoint(config=self.config)
        return confluence_checkpoint

    def __init__(
        self,
        config: ConfluenceConfig,
        secrets: dict,
        storage_path: str | None = None,
        entity_manager: Optional[EntityManager] = None,
    ):
        """Initialize Confluence source with configuration and credentials."""
        if storage_path is None:
            raise ValueError("storage_path must be provided")
        super().__init__(config, secrets, storage_path=storage_path, entity_manager=entity_manager)

        # Config is already validated via type hint and generic

        user = self.secrets.get("jira_user") or os.getenv("JIRA_USER")
        token = self.secrets.get("jira_api_token") or os.getenv("JIRA_API_TOKEN")

        if not user or not token:
            raise ValueError("Confluence credentials (JIRA_USER/JIRA_API_TOKEN) not found in secrets or environment")

        self.client = Confluence(
            url=self.config.url,
            username=user,
            password=token,
            cloud=True,  # Assuming cloud based on JIRA_CLOUD in env.example, but can be adjusted
        )

        self._connection_validated = False

    def _resolve_user_entity(self, user_obj: Any) -> Optional[str]:
        """Resolve a Confluence user object to a global entity ID."""
        if not self.entity_manager or not user_obj:
            return None

        # user_obj usually dict from 'by' field
        if not isinstance(user_obj, dict):
            return None

        user_data = user_obj
        account_id = user_data.get("accountId")
        # Could also be 'userKey' in older instances?
        # Cloud uses accountId.

        if not account_id:
            return None

        # Metadata
        name = user_data.get("displayName")
        email = user_data.get("email")  # Only if expanded and allowed

        # TODO:If email missing, and we really need it, we might fetch user?
        # For now, rely on accountId

        data = {
            "name": name,
            "display_name": name,
            "email": email,
            "type": user_data.get("type"),
        }

        try:
            entity = self.entity_manager.get_or_create_user(
                source_type="confluence", source_id=self.source_id, source_user_id=str(account_id), user_data=data
            )
            return entity.global_id
        except Exception as e:
            logger.warning(f"Failed to resolve entity for confluence user {account_id}: {e}")
            return None

    def _ensure_connection(self) -> None:
        """Validate connection to Confluence if not already validated."""
        if self._connection_validated:
            return

        try:
            # Atlassian Python API does not have a generic 'myself' for Confluence,
            # but we can try to fetch current user or a lightweight resource.
            # Using 'rest/api/user/current' is a standard way to check auth.
            self.client.get("rest/api/user/current")
            self._connection_validated = True
        except Exception as e:
            raise ValueError(f"Failed to authenticate with Confluence: {e}") from e

    def collect_documents_and_chunks(
        self,
        checkpoint: Checkpoint,
        update_checkpoint: bool = True,
        use_cached_data: bool = True,
        filters: Optional[Dict[str, list[str]]] = None,
    ) -> Tuple[List[DocumentUnit], List[ChunkRecord]]:
        """Collect documents and chunks from Confluence spaces."""
        self._ensure_connection()

        logger.info(f"Starting document collection for source {self.source_id}")
        if not isinstance(checkpoint, ConfluenceCheckpoint):
            checkpoint = ConfluenceCheckpoint(config=self.config, state=checkpoint.state if checkpoint else {})

        all_docs: List[DocumentUnit] = []
        all_chunks: List[ChunkRecord] = []

        for space in self.config.spaces:
            if filters and "space_ids" in filters and space not in filters["space_ids"]:
                logger.debug(f"Skipping space {space} due to filters")
                continue
            logger.info(f"Collecting documents for space: {space}")
            docs, chunks = self._collect_space(
                space, checkpoint, update_checkpoint=update_checkpoint, use_cached_data=use_cached_data
            )
            all_docs.extend(docs)
            all_chunks.extend(chunks)

        if update_checkpoint:
            try:
                base_dir = Path(self.storage_path) / self.source_id
                checkpoint.save(base_dir / "checkpoint.json")
            except Exception as e:
                logger.error(f"Failed to write checkpoint at end of run: {e}")

        return all_docs, all_chunks

    def _collect_space(
        self,
        space: str,
        checkpoint: ConfluenceCheckpoint,
        update_checkpoint: bool = True,
        use_cached_data: bool = True,
    ) -> Tuple[List[DocumentUnit], List[ChunkRecord]]:
        op_start = perf_counter()

        # Determine start time for query
        latest_ts = checkpoint.get_space_updated_ts(space)

        # Logic for initial fetch vs incremental
        # If checkpoint has a value that is NOT the default (meaning we ran before), use it.
        # If it IS the default, check if initial_lookback_days is set.
        # If initial_lookback_days is None, we want ALL history, so we don't filter by date.

        # However, Checkpoint logic defaults to 30 days if not set.
        # We need to distinguish "never ran" from "ran before".
        # We can check if space is in checkpoint.state["spaces"].

        is_incremental = space in checkpoint.state.get("spaces", {})
        logger.debug(f"Space {space}: incremental={is_incremental}, latest_ts={latest_ts}")

        cql_parts = [f'space = "{space}"', 'type = "page"']

        if is_incremental:
            # Incremental: fetch since last checkpoint
            since_dt = datetime.fromtimestamp(latest_ts, tz=timezone.utc)
            # Confluence CQL date format: "yyyy-MM-dd HH:mm" or "yyyy-MM-dd"
            # It seems strict. Let's try "yyyy-MM-dd".
            # Actually, for precision, we might need more. But CQL often supports "yyyy-MM-dd".
            # Let's use a safe format.
            updated_since = since_dt.strftime("%Y-%m-%d")
            cql_parts.append(f'lastModified >= "{updated_since}"')
        elif self.config.initial_lookback_days is not None:
            # First run with lookback limit
            since_dt = datetime.now(timezone.utc) - timedelta(days=self.config.initial_lookback_days)
            updated_since = since_dt.strftime("%Y-%m-%d")
            cql_parts.append(f'lastModified >= "{updated_since}"')
        # Else: fetch all (no date filter)

        cql = " AND ".join(cql_parts)
        cql += " ORDER BY lastModified ASC"

        logger.debug(f"Searching Confluence with CQL: {cql}")

        docs: List[DocumentUnit] = []
        chunks: List[ChunkRecord] = []
        max_updated_ts = latest_ts

        limit = ConfluenceSource.CONFLUENCE_API_LIMIT
        total_fetched = 0

        path = "rest/api/content/search"

        params = {
            "cql": cql,
            "limit": limit,
            "expand": "body.storage,version,history,metadata.labels,space,ancestors",
        }

        while True:
            call_start = perf_counter()
            try:
                results = self.client.get(path, params=params)
                status = "200"
            except Exception as e:
                status = "error"
                elapsed = perf_counter() - call_start
                API_CALLS.labels(source="confluence", source_id=self.source_id, method="cql", status=status).inc()
                API_LATENCY.labels(source="confluence", source_id=self.source_id, method="cql", status=status).observe(
                    elapsed
                )
                logger.error(f"Confluence API error: {e}")
                raise

            elapsed = perf_counter() - call_start
            API_CALLS.labels(source="confluence", source_id=self.source_id, method="cql", status=status).inc()
            API_LATENCY.labels(source="confluence", source_id=self.source_id, method="cql", status=status).observe(
                elapsed
            )

            if results is None:
                break
            results_list = results.get("results", [])  # client.get returns the full response, which has a 'results' key
            if not results_list:
                break

            logger.debug(f"Fetched {len(results_list)} pages (total: {total_fetched + len(results_list)})")

            for page in results_list:
                if "id" not in page:
                    continue

                doc, page_chunks = self._process_page(page, space)
                logger.debug(f"Processed page {page.get('id')} - {page.get('title')}")
                docs.append(doc)
                chunks.extend(page_chunks)

                # Track latest updated timestamp
                timestamp_str = page.get("version", {}).get("when")

                if timestamp_str:
                    try:
                        ts = self._parse_confluence_time(timestamp_str)
                        if ts and ts.timestamp() > max_updated_ts:
                            logger.debug(
                                f"Updating max_updated_ts for space {space}: {max_updated_ts} -> {ts.timestamp()}"
                            )
                            max_updated_ts = ts.timestamp()
                    except ValueError:
                        logger.warning(f"Could not parse updated timestamp: {timestamp_str}")

            total_fetched += len(results_list)

            if use_cached_data:
                self.__persist_api_messages(
                    space_key=space,
                    new_pages=results_list,
                    base_dir=Path(self.storage_path) / self.source_id,
                )

            if len(results_list) < limit:
                break

            # Pagination via _links.next
            next_link = results.get("_links", {}).get("next")
            if not next_link:
                break

            # Parse next link to update params
            parsed_next = urlparse(next_link)
            query_params = parse_qs(parsed_next.query)

            # Update params with values from next link (e.g. cursor, next start)
            # parse_qs returns list of values, we take the first one
            for k, v in query_params.items():
                if v:
                    params[k] = v[0]

            logger.debug(f"Fetching next page with params: {params}")

        op_elapsed = perf_counter() - op_start
        logger.info(f"Completed collection for space {space}: {total_fetched} pages in {op_elapsed:.3f}s")
        OP_LATENCY.labels(source="confluence", source_id=self.source_id, operation="collect_space").observe(op_elapsed)
        OP_ITEMS.labels(source="confluence", source_id=self.source_id, operation="collect_space").observe(total_fetched)

        if update_checkpoint:
            checkpoint.update_space_updated(space, max_updated_ts)

        return docs, chunks

    def __persist_api_messages(
        self,
        space_key: str,
        new_pages: List[dict],
        base_dir: Path,
    ) -> None:
        """Persist fetched pages to gzipped JSONL files."""
        if not new_pages:
            return

        space_dir = base_dir / space_key
        space_dir.mkdir(parents=True, exist_ok=True)

        # We'll store by page ID: "data/confluence/{source_id}/{space_key}/{page_id}.json.gz"
        # However, for Confluence, pages are distinct entities.
        # But that might be too many files.
        # TODO: Group by space like Jira? "data/confluence/{source_id}/{space_key}/{space_key}.jsonl.gz"
        for page in new_pages:
            page_id = page.get("id")
            if not page_id:
                continue

            out_path = space_dir / f"{page_id}.json.gz"
            try:
                with gzip.open(out_path, "wt", encoding="utf-8") as f:
                    json.dump(page, f, ensure_ascii=False)
            except Exception as e:
                logger.error(f"Failed to persist page {page_id}: {e}")

    def _process_page(self, page: Dict[str, Any], space_key: str) -> Tuple[DocumentUnit, List[ChunkRecord]]:
        page_id = page["id"]
        title = page["title"]
        web_ui = page.get("_links", {}).get("webui", "")
        uri = f"{self.config.url}{web_ui}" if web_ui else ""

        version_info = page.get("version", {})
        updated_str = version_info.get("when")
        author_info = version_info.get("by", {})
        author_name = author_info.get("displayName", "Unknown")

        # Created date is often in 'history'
        created_str = page.get("history", {}).get("createdDate")

        updated_at = self._parse_confluence_time(updated_str)
        created_at = self._parse_confluence_time(created_str) if created_str else None

        system_tags = [f"space:{space_key}", "type:page"]

        # Extract labels
        labels = page.get("metadata", {}).get("labels", {}).get("results", [])
        for label in labels:
            label_name = label.get("name")
            if label_name:
                system_tags.append(f"label:{label_name}")

        # Resolve Author Entity
        author_entity_id = self._resolve_user_entity(author_info)
        if author_entity_id:
            system_tags.append(f"user:{author_entity_id}")

        # Try to find parent_id from ancestors
        ancestors = page.get("ancestors", [])
        parent_id = None
        if ancestors:
            # The last ancestor is the direct parent
            parent_id = ancestors[-1].get("id")

        body_storage = page.get("body", {}).get("storage", {}).get("value", "")
        # Compute hash BEFORE markdown conversion if we want source truth,
        # but matching what Chunk does (if it did) is better.
        # Using body_storage (HTML) is stable.
        content_hash = DocumentUnit.compute_hash(body_storage.encode("utf-8")) if body_storage else None

        # Scan storage format for user mentions (moved here where body_storage is defined)
        # Format: <ri:user ri:accountId="ID" />
        if body_storage:
            import re

            # Extract accountId from ri:user tags
            mentions = re.findall(r'<ri:user\s+[^>]*ri:accountId="([^"]+)"', body_storage)

            for m_id in mentions:
                if len(m_id) > 5 and self.entity_manager:
                    try:
                        ent = self.entity_manager.get_user_by_source_id("confluence", self.source_id, m_id)
                        if ent:
                            system_tags.append(f"user:{ent.global_id}")
                    except Exception:
                        pass

        doc = DocumentUnit(
            document_id=f"confluence|{self.source_id}|{page_id}",
            source="confluence",
            source_doc_id=page_id,
            source_instance_id=self.source_id,
            uri=uri,
            title=title,
            author=author_name,
            parent_id=parent_id,
            language=None,  # Detection could be added here
            source_created_at=created_at,
            source_updated_at=updated_at,
            system_tags=system_tags,
            source_metadata={
                "page_id": page_id,
                "space": space_key,
                "title": title,
            },
            content_hash=content_hash,
        )

        chunks: List[ChunkRecord] = []

        markdown_text = self._html_to_markdown(body_storage)

        # Structural and Semantic Chunking
        chunk_data_list = self._chunk_text_by_headers_and_content(markdown_text, title)

        for chunk_i, (section_header, chunk_text) in enumerate(chunk_data_list):
            # Contextualize with header information
            # "Page: Title\nSection: Header\n\nContent..."
            final_text = f"Page: {title}\nSection: {section_header}\n\n{chunk_text}"

            # Tags
            chunk_tags = system_tags + ["type:content", f"section:{section_header}"]

            chunks.append(self._make_chunk(doc=doc, chunk_index=chunk_i, text=final_text, tags=chunk_tags))

        if not chunks:
            # Fallback if no text extracted?
            pass

        return doc, chunks

    def _chunk_text_by_headers_and_content(self, text: str, doc_title: str) -> List[Tuple[str, str]]:
        """Split text by headers and then by size constraints.

        Returns:
             List of (section_header, chunk_text) tuples.
        """
        if not text:
            return []

        chunks_out: List[Tuple[str, str]] = []

        # 1. Split by Headers (ATX style)
        # Capture the header line including hashes to know the level, though we just want the text
        # Regex: Start of line, 1-6 hashes, space, title
        parts = re.split(r"^(#{1,6} .+)$", text, flags=re.MULTILINE)

        # parts[0] is content before first header (Introduction)
        current_header = "Introduction"

        # Handle preamble
        if parts[0].strip():
            self._subchunk_content(current_header, parts[0], chunks_out)

        # Iterate pairs
        for i in range(1, len(parts), 2):
            header_line = parts[i].strip()  # e.g. "## My Header"
            content = parts[i + 1] if i + 1 < len(parts) else ""

            # Extract pure title from header line
            header_title = header_line.lstrip("#").strip()

            if (
                content.strip() or header_title
            ):  # If header exists, we might want it even if content is empty? No, only meaningful text.
                # Actually content might be empty but next subchunking handles it
                self._subchunk_content(header_title, content, chunks_out)

        return chunks_out

    def _subchunk_content(self, header: str, content: str, acc: List[Tuple[str, str]]) -> None:
        """Split content into chunks respecting size limits and code blocks."""
        content = content.strip()
        if not content:
            return

        limit = self.config.chunk_max_size_chars

        # If small enough, just add
        if len(content) <= limit:
            acc.append((header, content))
            return

        # Split by double newlines (paragraphs)
        paragraphs = re.split(r"\n{2,}", content)

        current_chunk_parts: List[str] = []
        current_len = 0

        in_code_block = False

        for p in paragraphs:
            # Check for code block toggles
            # Count triplets of backticks
            backtick_count = p.count("```")
            # If odd, we flipped state
            # (Simplification: assumes ``` is always a fence, not inline. Inline is usually single `)

            p_len = len(p)

            # Decision to flush
            # We flush if adding this paragraph exceeds limit AND we have something to flush
            # BUT we should NOT flush if we are inside a code block (try to keep code blocks together)
            # UNLESS the code block itself is huge (handled later/implicit?)

            if current_len + p_len > limit and current_chunk_parts:
                # If we are inside a code block, we arguably SHOULD flush if it's getting too big?
                # Or we try to keep it.
                # Let's say we prefer NOT to break code blocks.
                # But if we must (because it's huge), we must.

                # Basic strategy: Flush if full, unless inside code block and not absurdly huge.
                # Let's simple check: Flush if full.

                # Optimization: If inside code block, try to extend a bit?
                # Complexity tradeoff. Let's stick to size limit as soft rule, but maybe strict rule here.

                # Valid split point: if NOT inside code block.
                if not in_code_block:
                    # Flush
                    acc.append((header, "\n\n".join(current_chunk_parts)))
                    current_chunk_parts = []
                    current_len = 0

            current_chunk_parts.append(p)
            current_len += p_len

            if backtick_count % 2 != 0:
                in_code_block = not in_code_block

        if current_chunk_parts:
            acc.append((header, "\n\n".join(current_chunk_parts)))

    def _html_to_markdown(self, html: str) -> str:
        """Convert HTML to Markdown using markdownify."""
        if not html:
            return ""

        # Use markdownify with some custom options if needed,
        # but default is usually good enough for Confluence content.
        # We can strip tags that we don't want if needed.
        try:
            return str(md(html, heading_style="ATX")).strip()
        except RecursionError:
            logger.warning("RecursionError during markdown conversion. Falling back to plain text extraction.")
            # Fallback to plain text using BeautifulSoup
            from bs4 import BeautifulSoup

            return BeautifulSoup(html, "html.parser").get_text().strip()

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

    def _parse_confluence_time(self, time_str: str) -> Optional[datetime]:
        if not time_str:
            return None
        # Try common formats
        formats = ["%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d"]
        for fmt in formats:
            try:
                return datetime.strptime(time_str, fmt)
            except ValueError:
                continue
        return None

    def collect_cached_documents_and_chunks(
        self,
        filters: Optional[Dict[str, list[str]]] = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> Tuple[List[DocumentUnit], List[ChunkRecord]]:
        """Rehydrate documents and chunks exclusively from cached data."""
        base_dir = Path(self.storage_path) / self.source_id
        if not base_dir.exists():
            logger.warning(f"Cache directory {base_dir} does not exist")
            return [], []

        logger.info(f"Collecting cached documents from {base_dir}")
        all_docs: List[DocumentUnit] = []
        all_chunks: List[ChunkRecord] = []

        if filters and "space_ids" in filters:
            target_spaces = sorted(set(filters["space_ids"]))
        else:
            target_spaces = [entry.name for entry in base_dir.iterdir() if entry.is_dir()]

        logger.debug(f"Target spaces for cached collection: {target_spaces}")

        for space in target_spaces:
            space_dir = base_dir / space
            if not space_dir.exists():
                continue

            # Iterate over all .json.gz files in space_dir
            for entry in space_dir.glob("*.json.gz"):
                logger.debug(f"Loading cached file: {entry}")
                try:
                    with gzip.open(entry, "rt", encoding="utf-8") as f:
                        page = json.load(f)

                        if "id" not in page:
                            continue

                        # Date filtering
                        updated_str = page.get("version", {}).get("when")
                        updated_dt = self._parse_confluence_time(updated_str)

                        if updated_dt:
                            if date_from and updated_dt < date_from:
                                continue
                            if date_to and updated_dt > date_to:
                                continue

                        doc, chunks = self._process_page(page, space)
                        all_docs.append(doc)
                        all_chunks.extend(chunks)
                except Exception as e:
                    logger.error(f"Failed to read cache file {entry}: {e}")

        return all_docs, all_chunks


def main() -> None:
    """Execute Confluence document collection."""
    logging.basicConfig(level=logging.INFO)

    # Example configuration from environment
    confluence_url = os.getenv("JIRA_INSTANCE_URL", "")
    confluence_spaces = os.getenv("CONFLUENCE_SPACES", "").split(",") if os.getenv("CONFLUENCE_SPACES") else []
    storage_path = "./data/confluence"
    source_id = "confluence-main"
    jira_user = os.getenv("JIRA_USER")
    jira_api_token = os.getenv("JIRA_API_TOKEN")

    if not confluence_url or not jira_user:
        print("Please set JIRA_INSTANCE_URL, JIRA_USER, and JIRA_API_TOKEN")
        return

    secrets = {"jira_user": jira_user, "jira_api_token": jira_api_token}

    config = ConfluenceConfig(
        id=source_id,
        url=confluence_url,
        spaces=confluence_spaces,
        use_cached_data=False,
        initial_lookback_days=None,
    )

    entity_manager = EntityManager(storage_path="data/entities.json")
    source = ConfluenceSource(config, secrets, storage_path=storage_path, entity_manager=entity_manager)

    # Load checkpoint
    checkpoint_dir = Path(storage_path) / source_id
    checkpoint_path = checkpoint_dir / "checkpoint.json"

    if checkpoint_path.exists():
        try:
            checkpoint = ConfluenceCheckpoint.load(checkpoint_path, source.config)
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {checkpoint_path}: {e}")
            checkpoint = ConfluenceCheckpoint(config=source.config)
    else:
        checkpoint = ConfluenceCheckpoint(config=source.config)

    print("Starting collection...")
    docs, chunks = source.collect_documents_and_chunks(checkpoint)
    print(f"Collected {len(docs)} documents and {len(chunks)} chunks.")


if __name__ == "__main__":
    main()
