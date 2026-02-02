"""Slack source: conversations and messages collection with adaptive rate limiting and metrics.

This module implements a Slack data source that:
- Uses Slack WebClient with retry handlers (respecting Retry-After).
- Applies an adaptive, tier-aware rate limiter per method.
- Emits Prometheus metrics for per-call latency/count and per-operation totals.
- Logs operation lifecycle at INFO and per-page details at DEBUG.

The design follows Slack rate limiting guidance:
https://docs.slack.dev/apis/web-api/rate-limits/
"""

import gzip
import json
import logging
import os
import re
import time
from dataclasses import field
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Final, List, Optional, Set, Tuple, cast

from pydantic import Field
from sentence_transformers.SentenceTransformer import SentenceTransformer
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slack_sdk.http_retry.builtin_handlers import (
    ConnectionErrorRetryHandler,
    RateLimitErrorRetryHandler,
    ServerErrorRetryHandler,
)

from src.entities.manager import EntityManager

# Import Prometheus metrics from centralized module
from src.metrics.metrics import API_CALLS, API_LATENCY, OP_ITEMS, OP_LATENCY, USER_CACHE_HITS, USER_CACHE_MISSES
from src.models.documents import ChunkRecord, DocumentUnit
from src.models.source_config import BaseSourceConfig
from src.sources.checkpoint import Checkpoint
from src.sources.ratelimiter import AdaptiveRateLimiter
from src.sources.source import Source
from src.utils.persistent_cache import PersistentCache


class SlackConfig(BaseSourceConfig):
    """Configuration for Slack source."""

    channel_window_minutes: int = Field(default=60, description="Window size in minutes for channel chunking")
    workspace_domain: str = Field(..., description="Workspace domain for constructing URLs")
    user_cache_path: Optional[str] = Field(default=None, description="Path to the user cache file")
    user_cache_ttl_seconds: int = Field(default=7 * 24 * 60 * 60, description="TTL for user cache in seconds")
    tier2_rpm: float = Field(default=18.0, description="Target RPM for Tier 2 methods")
    tier2_cap: float = Field(default=24.0, description="Max RPM for Tier 2 methods")
    tier3_rpm: float = Field(default=45.0, description="Target RPM for Tier 3 methods")
    tier3_cap: float = Field(default=60.0, description="Max RPM for Tier 3 methods")
    initial_lookback_days: int = Field(default=7, description="Initial lookback days for checkpoint")
    channel_types: List[str] = Field(
        default_factory=lambda: ["public_channel", "private_channel", "im", "mpim"],
        description="List of channel types to fetch",
    )
    chunk_time_interval_minutes: int = Field(default=240, description="Max minutes between messages in a chunk")
    chunk_max_size_chars: int = Field(default=2000, description="Max characters per chunk")
    chunk_max_count: int = Field(default=20, description="Max messages per chunk")
    chunk_similarity_threshold: float = Field(default=0.15, description="Cosine similarity threshold for splitting")
    chunk_embedding_model: str = Field(default="all-MiniLM-L6-v2", description="Model for semantic embeddings")
    api_retries: int = Field(default=3, description="Number of retries for unexpected API errors")


class SlackCheckpoint(Checkpoint[SlackConfig]):
    """Checkpoint specifically for Slack source."""

    config: SlackConfig
    state: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize checkpoint state with default values."""
        self.state.setdefault("channels", {})
        default_latest_ts = self.__get_default_latest_ts()
        self.state.setdefault("default_latest_ts", float(default_latest_ts))

    def __get_default_latest_ts(self) -> float:
        return (datetime.now(timezone.utc) - timedelta(days=self.config.initial_lookback_days)).timestamp()

    def get_latest_ts(self, channel_id: str) -> float:
        """Get the latest timestamp for a channel."""
        channels: Dict[str, Any] = self.state.setdefault("channels", {})
        value = channels.get(channel_id)
        default_latest_ts = self.state.get("default_latest_ts", self.__get_default_latest_ts())
        if value is not None and isinstance(value, dict) and value.get("latest_ts") is not None:
            return float(value.get("latest_ts"))
        return float(default_latest_ts)

    def get_earliest_ts(self, channel_id: str) -> Optional[float]:
        """Get the earliest timestamp for a channel."""
        channels: Dict[str, Any] = self.state.setdefault("channels", {})
        value = channels.get(channel_id)
        if value is not None and isinstance(value, dict) and value.get("earliest_ts") is not None:
            return float(value.get("earliest_ts"))
        return None

    def update_channel_ts(
        self,
        channel_id: str,
        newest_ts: Optional[float] = None,
        oldest_ts: Optional[float] = None,
    ) -> None:
        """Update the latest and earliest timestamps for a channel."""
        channels: Dict[str, Any] = self.state.setdefault("channels", {})
        channel_data = channels.setdefault(channel_id, {})

        if newest_ts is not None:
            current_latest = channel_data.get("latest_ts")
            if current_latest is None or newest_ts > float(current_latest):
                channel_data["latest_ts"] = float(newest_ts)

        if oldest_ts is not None:
            current_earliest = channel_data.get("earliest_ts")
            if current_earliest is None or oldest_ts < float(current_earliest):
                channel_data["earliest_ts"] = float(oldest_ts)


class SlackSource(Source[SlackConfig]):
    """Slack source implementation with proactive rate limiting and metrics.

    Initializes a Slack `WebClient` with built-in retry handlers, wires an
    adaptive per-method limiter for `conversations.list` (Tier 2) and
    `conversations.history` (Tier 3), and instruments calls and operations
    with Prometheus histograms and counters.

    Config keys:
        - channel_types: list[str] of conversation types to fetch (e.g., public_channel).
        - initial_lookback_days: int days for default_latest_ts if state empty.
        - tier2_rpm, tier2_cap: optional overrides for Tier 2 pacing.
        - tier3_rpm, tier3_cap: optional overrides for Tier 3 pacing.

    Secrets:
        - bot_token: Slack bot OAuth token.

    Todo:
    - Add support for hierarchical chunking
    - Add support for optional enrichment (NER/Pii) for tags at chunk or document level.
    - Add support to retrieve and update edited messages.
    """

    SLACK_API_LIMIT: Final[int] = 200
    """Max messages per API call to Slack"""

    @classmethod
    def create(
        cls,
        config: SlackConfig,
        data_dir: str,
        secrets: Optional[dict] = None,
        entity_manager: Optional[EntityManager] = None,
    ) -> "SlackSource":
        """Create a SlackSource instance with the given configuration."""
        # Pass storage_path explicitly to constructor
        storage_path = os.path.join(data_dir, "slack")

        # Ensure user cache path is set relative to data_dir if not explicit
        if not config.user_cache_path:
            config.user_cache_path = os.path.join(storage_path, config.id, "slack_users_cache.json")

        # Secrets handling
        if secrets is None:
            secrets = {}
        if "bot_token" not in secrets:
            token = os.getenv("SLACK_BOT_TOKEN")
            if token:
                secrets["bot_token"] = token

        return cls(config=config, secrets=secrets, storage_path=storage_path, entity_manager=entity_manager)

    def __init__(
        self,
        config: SlackConfig,
        secrets: dict,
        storage_path: str | None = None,
        cache: Optional[PersistentCache] = None,
        rate_limiter: Optional[AdaptiveRateLimiter] = None,
        entity_manager: Optional[EntityManager] = None,
    ):
        """Initialize Slack source with configuration, credentials, and rate limiting."""
        if storage_path is None:
            raise ValueError("storage_path must be provided")
        super().__init__(config, secrets, storage_path=storage_path, entity_manager=entity_manager)
        if not self.secrets["bot_token"]:
            raise ValueError("Slack bot token is not set")

        # Config is already validated via type hint and generic
        # but we can ensure it's the right type if needed, though type checker handles it.

        # Ensure defaults for paths if not provided
        if not self.config.user_cache_path:
            # Default to data/slack/slack-main/slack_users_cache.json if not specified
            # But here we don't know "data" dir location easily unless passed.
            # We rely on caller to set it or we default to relative.
            self.config.user_cache_path = os.path.join(storage_path, config.id, "slack_users_cache.json")

        # User cache config (JSON persistent cache)
        if cache:
            self._user_cache = cache
        else:
            path = Path(self.config.user_cache_path).expanduser().resolve()
            parent = path.parent
            if not parent.exists():
                try:
                    parent.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    logger.warning(f"Could not create user cache directory {parent}: {e}")

            self._user_cache = PersistentCache(
                path=str(path),
                ttl_seconds=self.config.user_cache_ttl_seconds,
            )

        # Initialize WebClient with retry handlers that respect Retry-After
        self.client = WebClient(
            token=self.secrets["bot_token"],
            retry_handlers=[
                RateLimitErrorRetryHandler(max_retry_count=5),
                ServerErrorRetryHandler(max_retry_count=2),
                ConnectionErrorRetryHandler(max_retry_count=2),
            ],
        )

        # Initialize adaptive rate limiter (Tier-aware)
        if rate_limiter:
            self.rate_limiter = rate_limiter
        else:
            self.rate_limiter = AdaptiveRateLimiter(
                defaults={
                    "conversations.list": {
                        "rpm": self.config.tier2_rpm,
                        "cap": self.config.tier2_cap,
                        "burst": 5,
                    },
                    "conversations.history": {
                        "rpm": self.config.tier3_rpm,
                        "cap": self.config.tier3_cap,
                        "burst": 10,
                    },
                    "conversations.info": {
                        "rpm": self.config.tier3_rpm,
                        "cap": self.config.tier3_cap,
                        "burst": 10,
                    },
                    "users.info": {
                        "rpm": self.config.tier2_rpm,
                        "cap": self.config.tier2_cap,
                        "burst": 5,
                    },
                }
            )

        # Lazy loaded embedding model
        self._embedding_model: SentenceTransformer | None = None

    def _api_call(self, method: str, func: Callable, **kwargs: Any) -> Any:
        """Execute a Slack API call with standard rate limiting, retries, and metrics.

        Args:
            method: The method name for metrics and rate limiting (e.g., "users.info").
            func: The API client function to call.
            **kwargs: Arguments to pass to the function.

        Returns:
            The API response.

        Raises:
            SlackApiError: If the API call fails with a 4xx error or after retrying 5xx errors.
            Exception: If an unexpected error occurs after retries.
        """
        consecutive_errors = 0
        while True:
            self.rate_limiter.acquire(method)
            call_start = perf_counter()
            try:
                resp = func(**kwargs)
                consecutive_errors = 0

                # Success metrics
                status = str(getattr(resp, "status_code", 200))
                API_CALLS.labels(source="slack", source_id=self.source_id, method=method, status=status).inc()
                API_LATENCY.labels(source="slack", source_id=self.source_id, method=method, status=status).observe(
                    perf_counter() - call_start
                )
                return resp

            except SlackApiError as e:
                # Rate limiting management
                status = str(getattr(getattr(e, "response", None), "status_code", 429))
                API_CALLS.labels(source="slack", source_id=self.source_id, method=method, status=status).inc()
                API_LATENCY.labels(source="slack", source_id=self.source_id, method=method, status=status).observe(
                    perf_counter() - call_start
                )

                if getattr(e, "response", None) is not None and getattr(e.response, "status_code", None) == 429:
                    wait_seconds = int(e.response.headers.get("Retry-After", "1"))
                    logger.info(f"429 on {method}, Retry-After={wait_seconds}s")
                    self.rate_limiter.on_rate_limited(method, wait_seconds)
                    time.sleep(wait_seconds)
                    continue

                # Check for non-recoverable client errors (4xx excluding 429)
                status_int = int(status)
                if 400 <= status_int < 500:
                    logger.debug(f"Client error in {method}: {e}")
                    raise

                # Handle recoverable server/unknown errors (5xx)
                consecutive_errors += 1
                if consecutive_errors <= self.config.api_retries:
                    wait_seconds = 2**consecutive_errors
                    logger.warning(
                        f"Server error {status} from Slack API in {method} (attempt {consecutive_errors}/{self.config.api_retries}). Retrying in {wait_seconds}s..."
                    )
                    time.sleep(wait_seconds)
                    continue

                raise
            except Exception as e:
                logger.error(f"Unexpected error in {method}: {e}")
                consecutive_errors += 1
                if consecutive_errors <= self.config.api_retries:
                    wait_seconds = 2**consecutive_errors
                    logger.warning(
                        f"Retrying {method} after unexpected error (attempt {consecutive_errors}/{self.config.api_retries}) in {wait_seconds}s..."
                    )
                    time.sleep(wait_seconds)
                    continue
                raise

    def load_checkpoint(self) -> SlackCheckpoint:
        """Load existing checkpoint or create a new one."""
        from pathlib import Path

        cp_path = Path(self.storage_path) / self.source_id / "checkpoint.json"
        if cp_path.exists():
            return SlackCheckpoint.load(cp_path, self.config)
        return SlackCheckpoint(config=self.config)

    def _list_conversations(self, channel_types: List[str]) -> List[dict]:
        """List Slack conversations using cursored pagination.

        Applies proactive rate limiting, collects per-call metrics, and
        summarizes total items/time as an operation metric.

        Returns:
            List[dict]: Accumulated conversation objects from Slack API.
        """
        op_start = perf_counter()
        op_items = 0
        # Build types as comma-separated string if provided as list
        types_str = ",".join(channel_types) if isinstance(channel_types, list) else channel_types
        logger.debug(f"list_conversations: start types={types_str}")
        conversations: list[dict] = []
        cursor = None
        while True:
            # Call API using wrapper
            try:
                resp = self._api_call(
                    "conversations.list",
                    self.client.conversations_list,
                    cursor=cursor,
                    limit=SlackSource.SLACK_API_LIMIT,
                    types=types_str,
                    exclude_archived=True,
                )
            except Exception as e:
                logger.error(
                    f"Error listing conversations (conversation list process interrupted, not all channels collected): {e}"
                )
                break

            logger.debug(f"list_conversations: page channels={len(resp.get('channels', []))}")
            page_items = len(resp.get("channels", []))
            op_items += page_items
            conversations.extend(resp.get("channels", []))
            resp_metadata: dict = resp.get("response_metadata", {})
            cursor = resp_metadata.get("next_cursor")
            if not cursor:
                break
        # Collect metrics
        op_elapsed = perf_counter() - op_start
        logger.info(f"list_conversations: done items={op_items} elapsed={op_elapsed:.3f}s")
        OP_LATENCY.labels(source="slack", source_id=self.source_id, operation="list_conversations").observe(op_elapsed)
        OP_ITEMS.labels(source="slack", source_id=self.source_id, operation="list_conversations").observe(op_items)
        return conversations

    def _persist_channel_metadata(self, channel_dir: Path, channel: dict) -> None:
        """Persist the raw channel metadata returned by Slack to metadata.json.

        Args:
            channel_dir: The directory to persist the channel metadata to.
            channel: A dictionary representing the channel to persist.

        Returns:
            None
        """
        channel_dir.mkdir(parents=True, exist_ok=True)
        meta_path = channel_dir / "metadata.json"
        try:
            with meta_path.open("w", encoding="utf-8") as handle:
                json.dump(channel, handle, ensure_ascii=False, indent=2)
            logger.debug(f"Stored channel metadata at {meta_path}")
        except Exception as err:
            logger.error(f"Failed to write channel metadata {meta_path}: {err}")

    def _load_channel_metadata(self, channel_dir: Path, channel_id: str) -> dict:
        """Load channel metadata from metadata.json, falling back to basic info.

        Args:
            channel_dir: The directory to load the channel metadata from.
            channel_id: The ID of the channel to load the metadata for.

        Returns:
            A dictionary representing the channel metadata.
        """
        meta_path = channel_dir / "metadata.json"
        if meta_path.exists():
            try:
                with meta_path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
                    if isinstance(data, dict):
                        return data
            except Exception as err:
                logger.warning(f"Failed to read metadata for {channel_id}: {err}")
        logger.warning(f"Using fallback metadata for channel {channel_id}")
        return {"id": channel_id, "name": channel_id}

    def _get_channel_metadata(self, channel_id: str, use_cache: bool = True) -> dict:
        """Retrieve channel metadata from Slack API or cache.

        Args:
            channel_id: The ID of the channel to retrieve metadata for.
            use_cache: If True, attempt to load from local cache first.

        Returns:
            A dictionary representing the channel metadata.
        """
        base_dir = Path(self.storage_path) / self.source_id
        channel_dir = base_dir / channel_id

        if use_cache:
            meta_path = channel_dir / "metadata.json"
            if meta_path.exists():
                try:
                    return self._load_channel_metadata(channel_dir, channel_id)
                except Exception:
                    pass  # Fallback to API

        # Fetch from API
        try:
            resp = self._api_call("conversations.info", self.client.conversations_info, channel=channel_id)
            channel: dict[str, dict] = resp.get("channel", {})
            self._persist_channel_metadata(channel_dir, channel)
            return channel
        except Exception:
            logger.warning(f"Using fallback metadata for channel {channel_id} due to persistent errors")
            return {"id": channel_id, "name": channel_id}

    def _list_messages(
        self,
        channel_id: str,
        checkpoint: SlackCheckpoint,
        latest_ts_override: Optional[float] = None,
    ) -> List[dict]:
        """List messages for a given channel using cursored pagination.

        Applies proactive rate limiting and collects both per-call and
        per-operation metrics. Honors `oldest_ts` from the checkpoint to
        enable incremental collection.

        Args:
            channel_id: Slack channel/conversation ID.
            checkpoint: Checkpoint containing `oldest_ts` for incremental sync.
            latest_ts_override: Optional override for the latest timestamp from checkpoint.

        Returns:
            List[dict]: Accumulated message objects for the channel.
        """
        op_start = perf_counter()
        op_items = 0
        logger.debug(f"list_messages: start channel={channel_id}")
        messages: list[dict] = []
        cursor = None
        while True:
            # Get latest timestamp from checkpoint or override if provided
            latest_ts = latest_ts_override if latest_ts_override is not None else checkpoint.get_latest_ts(channel_id)
            latest_ts_str = str(float(latest_ts)) if latest_ts is not None else None

            # Call API using wrapper
            resp = self._api_call(
                "conversations.history",
                self.client.conversations_history,
                cursor=cursor,
                channel=channel_id,
                limit=SlackSource.SLACK_API_LIMIT,
                oldest=latest_ts_str,
            )

            page_msgs: List[Dict[str, Any]] = resp.get("messages", [])
            messages.extend(page_msgs)
            page_items = len(page_msgs)
            logger.debug(f"list_messages: page channel={channel_id} messages={page_items}")
            op_items += page_items
            resp_metadata: dict = resp.get("response_metadata", {})
            cursor = resp_metadata.get("next_cursor")
            if not cursor:
                break
        # Collect metrics
        op_elapsed = perf_counter() - op_start
        logger.info(f"list_messages: done channel={channel_id} items={op_items} elapsed={op_elapsed:.3f}s")
        OP_LATENCY.labels(source="slack", source_id=self.source_id, operation="list_messages").observe(op_elapsed)
        OP_ITEMS.labels(source="slack", source_id=self.source_id, operation="list_messages").observe(op_items)
        return messages

    def _get_user_info(self, user_id: str, force_refresh: bool = False) -> dict:
        """Fetch Slack user info using users.info with persistent cache.

        Args:
            user_id: Slack user ID (e.g., 'U12345678').
            force_refresh: If True, bypass cache and refresh from API.

        Returns:
            dict: Compact user info object.
        """
        entry = None if force_refresh else self._user_cache.get(user_id)
        compact = None
        fetched_from_api = False

        if entry and isinstance(entry, dict):
            USER_CACHE_HITS.inc()
            logger.debug(f"user cache HIT for {user_id}")
            compact = cast(Dict[str, Any], entry)
        else:
            USER_CACHE_MISSES.inc()
            logger.debug(f"user cache MISS for {user_id}")

            resp = self._api_call("users.info", self.client.users_info, user=user_id)

            user = resp.get("user", {}) if isinstance(resp, dict) else getattr(resp, "data", {}).get("user", {})
            # Build compact user info
            compact = {
                "id": user.get("id"),
                "team_id": user.get("team_id"),
                "name": user.get("name"),
                "real_name": user.get("real_name"),
                "profile": user.get("profile", {}),
                "is_bot": user.get("is_bot"),
                "updated": user.get("updated"),
                "cached_at": int(time.time()),
            }
            fetched_from_api = True

        # Register with EntityManager
        updated_entity = False
        if self.entity_manager and "global_entity_id" not in compact:
            try:
                profile = compact.get("profile", {})
                if not isinstance(profile, dict):
                    profile = {}

                user_data = {
                    "name": compact.get("name")
                    or compact.get("real_name")
                    or profile.get("real_name"),  # Fallback to real_name if name is missing
                    "real_name": compact.get("real_name") or profile.get("real_name"),
                    "email": profile.get("email"),
                    "display_name": profile.get("display_name"),
                    "title": profile.get("title"),
                }
                if self.entity_manager:
                    entity = self.entity_manager.get_or_create_user(
                        source_type="slack", source_id=self.source_id, source_user_id=user_id, user_data=user_data
                    )
                    compact["global_entity_id"] = entity.global_id
                    updated_entity = True
            except Exception as e:
                logger.error(f"Failed to register user {user_id} with EntityManager: {e}")

        if fetched_from_api or updated_entity:
            self._user_cache.set(user_id, compact)

        return compact

    def _extract_threads(self, channel: dict, messages: List[dict]) -> dict:
        """Extract thread structures from a channel's messages.

        Builds a lightweight representation of channel container and per-thread
        message groupings. This method does not compute timestamps or update
        checkpoints; it only groups messages by thread context.

        Args:
            channel: Slack channel object (must include 'id'; 'name' optional).
            messages: List of message dicts from conversations.history.

        Returns:
            dict: Dictionary of thread-like entries. Each entry includes:
                - Key:
                  - channel id or thread_ts as key
                - Value:
                    - is_channel: True if container for channel messages
                    - is_thread: True if represents a thread group
                    - has_parent: Only for threads, True if parent message observed
                    - channel_id: For threads, the channel id they belong to
                    - messages: List of messages, most-recent-first (insert at 0)
        """
        threads_local: dict = {}
        if len(messages) > 0:
            threads_local[channel["id"]] = {
                "is_channel": True,
                "is_thread": False,
                "has_parent": False,
                "messages": [],
            }
            for message in messages:
                try:
                    # If message is a thread, add it to the thread structure
                    if "thread_ts" in message and message["thread_ts"] is not None:
                        tkey = message["thread_ts"]
                        # If thread already exists, add message to it
                        if tkey in threads_local:
                            threads_local[tkey]["messages"].insert(0, message)
                        # If thread does not exist, create it
                        else:
                            threads_local[tkey] = {
                                "is_channel": False,
                                "has_parent": False,
                                "is_thread": True,
                                "channel_id": channel["id"],
                                "id": message["thread_ts"],
                                "messages": [message],
                            }
                        # Mark parent if present and mirror into channel container (thread container)
                        if message["thread_ts"] == message.get("ts"):
                            threads_local[tkey]["has_parent"] = True
                            threads_local[channel["id"]]["messages"].insert(0, message)
                        # Broadcasts appear also in channel timeline (thread container)
                        if "subtype" in message and message["subtype"] == "thread_broadcast":
                            threads_local[channel["id"]]["messages"].insert(0, message)
                    else:
                        # Non-thread messages go to channel container (channel container)
                        threads_local[channel["id"]]["messages"].insert(0, message)
                except Exception as e:
                    logger.error(f"Error extracting threads: {e} for message {message}")
        return threads_local

    def _extract_document_from_thread(
        self, channel: dict, thread: dict
    ) -> List[Tuple[DocumentUnit, List[ChunkRecord]]]:
        """Build DocumentUnit(s) from a thread/channel entry.

        Threads become a single document. Channel containers are bucketed into
        N-minute windows (config `channel_window_minutes`) producing one
        document per window. The document URI uses a Slack fallback URL built
        from the first message timestamp in the document if `workspace_domain`
        is configured.

        Args:
            channel: Slack channel dict (expects 'id', optional 'name').
            thread: Entry from __extract_threads for this channel.

        Returns:
            List[Tuple[DocumentUnit, List[ChunkRecord]]]: Pairs of document and
            the chunks that belong to that document.
        """
        ch_name = channel.get("name") or channel["id"]
        docs_with_msgs: List[Tuple[DocumentUnit, List[ChunkRecord]]] = []

        # Document based on a thread messages
        if thread.get("is_thread"):
            msgs = thread.get("messages", [])
            # Skip creating a document when there are no messages (no chunks)
            if not msgs:
                return docs_with_msgs
            thread_ts = thread.get("id")
            if not thread_ts:
                raise ValueError("Thread timestamp is missing")
            return self.__process_thread_messages(channel, ch_name, msgs, thread_ts)
        else:
            # Document based on a windowed channel messages
            window_minutes = int(self.config.channel_window_minutes)
            msgs = sorted(thread.get("messages", []), key=lambda m: float(m.get("ts", 0.0)))
            return self.__process_windowed_messages(channel, ch_name, msgs, window_minutes)

    def get_specific_query(self, document_ids: List[str]) -> Optional[Dict[str, Any]]:
        """Return a Qdrant filter query if any of the document IDs match resources in this source.

        Args:
            document_ids: List of specific document identifiers.

        Returns:
            Optional[Dict[str, Any]]: A Qdrant filter dictionary (e.g. {"must": [...]}) if a
            match is found, or None if no specific identifier is detected.
        """
        # For Slack, we might match channel IDs or message TS, but currently we have no specific logic.
        # We must align with the base class signature.
        return None

    # --------- Chunking helpers ----------
    def _resolve_user_display(self, user_id: Optional[str]) -> str:
        """Resolve a user display name from a user id.

        Args:
            user_id: A string representing a Slack user id.

        Returns:
            A string representing the resolved user display name.
        """
        if not user_id:
            return "unknown"
        try:
            info = self._get_user_info(user_id)
            return info.get("real_name") or info.get("name") or user_id
        except Exception:
            return user_id

    def _replace_mentions(self, text: str) -> str:
        """Replace mentions in a text with the user display name.

        Args:
            text: A string representing a Slack message.

        Returns:
            A string representing the text with mentions replaced with the user display name.
        """
        import re

        def repl(m: re.Match) -> str:
            uid = m.group(1)
            return f"@{self._resolve_user_display(uid)}"

        return re.sub(r"<@([A-Z0-9]+)>", repl, text or "")

    def _extract_simple_tags(self, text: str) -> List[str]:
        """Extract simple tags from a text.

        Args:
            text: A string representing a Slack message.

        Returns:
            A list of strings representing the simple tags extracted from the text.
        """
        import re
        import urllib.parse

        tags: set[str] = set()
        url_re = re.compile(r"(https?://[^\s>]+)")
        for u in url_re.findall(text or ""):
            try:
                p = urllib.parse.urlparse(u)
                if p.netloc:
                    tags.add(f"url:{u}")
                    tags.add(f"domain:{p.netloc.lower()}")
            except Exception:
                pass
        email_re = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}")
        for e in email_re.findall(text or ""):
            el = e.lower()
            tags.add(f"email:{el}")
            dom = el.split("@", 1)[-1]
            tags.add(f"email_domain:{dom}")
        return list(tags)[:10]

    def _format_message_for_chunk(self, msg: dict) -> str:
        """Format a message for a chunk.

        Args:
            msg: A dictionary representing a Slack message.

        Returns:
            A string representing the formatted message for a chunk.
        """
        ts = float(msg.get("ts", 0.0)) if msg.get("ts") else 0.0
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        user_disp = self._resolve_user_display(msg.get("user"))
        text = self._replace_mentions(msg.get("text", ""))
        return f"[{dt.strftime('%Y-%m-%d %H:%M %Z')}] {user_disp}: {text}"

    def _make_chunk(
        self,
        doc: DocumentUnit,
        chunk_index: int,
        text: str,
        tags: List[str],
        user_tags: Optional[List[str]] = None,
        created: Optional[datetime] = None,
        updated: Optional[datetime] = None,
    ) -> ChunkRecord:
        """Create a flat ChunkRecord for a given document and message text.

        Args:
            doc: A DocumentUnit representing the document the chunk belongs to.
            chunk_index: An integer representing the index of the chunk in the document.
            text: A string representing the text of the chunk.
            tags: A list of strings representing the tags of the chunk.
            user_tags: A list of user-specific tags (e.g., user:<id>).

        Returns:
            A ChunkRecord representing the chunk.
        """
        content_hash = DocumentUnit.compute_hash(text.encode("utf-8"))
        chunk_id = sha256(f"{doc.document_id}|{chunk_index}|{content_hash}".encode()).hexdigest()
        return ChunkRecord(
            chunk_id=chunk_id,
            parent_document_id=doc.document_id,
            parent_chunk_id=None,
            level=0,
            chunk_index=chunk_index,
            text=text,
            language=None,
            mime_type="text/plain",
            page=None,
            start_line=None,
            end_line=None,
            start_char=None,
            end_char=None,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            system_tags=tags,
            user_tags=user_tags or [],
            chunking_strategy="slack_message_line_v1",
            chunk_overlap=0,
            content_hash=content_hash,
        )

    def _get_embedding_model(self) -> SentenceTransformer:
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(self.config.chunk_embedding_model)
        return self._embedding_model

    def _collect_chunk_tags(self, messages: List[dict]) -> Tuple[Set[str], Set[str]]:
        """Collect tags for a batch of messages.

        Args:
            messages: A list of message dictionaries.

        Returns:
            A tuple containing a set of general tags and a set of user tags.
        """
        all_tags: Set[str] = set()
        user_tags: Set[str] = set()
        mention_re = re.compile(r"<@(U[A-Z0-9]+)>")

        for cm in messages:
            all_tags.update(self._extract_simple_tags(cm.get("text", "")))

            # Author tagging
            uid = cm.get("user")
            if uid:
                self._add_user_tag(uid, user_tags)

            # Mention tagging
            mentions = mention_re.findall(cm.get("text", ""))
            for mentioned_uid in mentions:
                self._add_user_tag(mentioned_uid, user_tags)

        return all_tags, user_tags

    def _add_user_tag(self, uid: str, user_tags: Set[str]) -> None:
        """Helper to resolve user ID and add tag."""
        try:
            u_info = self._get_user_info(uid)
            if "global_entity_id" in u_info:
                user_tags.add(f"user:{u_info['global_entity_id']}")
        except Exception:
            pass

    def _finalize_chunk(self, doc: DocumentUnit, start_idx: int, messages: List[dict]) -> ChunkRecord:
        """Create a ChunkRecord from a batch of messages."""
        chunk_text = "\n".join([self._format_message_for_chunk(cm) for cm in messages])
        all_tags, user_tags = self._collect_chunk_tags(messages)

        # Determine created/updated from messages in this chunk
        chunk_created = None
        chunk_updated = None
        if messages:
            try:
                c_ts_first = float(messages[0].get("ts", 0))
                c_ts_last = float(messages[-1].get("ts", 0))
                if c_ts_first > 0:
                    chunk_created = datetime.fromtimestamp(c_ts_first, tz=timezone.utc)
                if c_ts_last > 0:
                    chunk_updated = datetime.fromtimestamp(c_ts_last, tz=timezone.utc)
            except Exception:
                pass

        return self._make_chunk(
            doc,
            start_idx,
            chunk_text,
            list(all_tags),
            user_tags=list(user_tags),
            created=chunk_created,
            updated=chunk_updated,
        )

    def _build_chunks_for_messages(self, doc: DocumentUnit, msgs: List[dict]) -> List[ChunkRecord]:
        """Build flat chunks for a list of Slack message dicts with multi-strategy splitting.

        Strategies:
        1. Time: Split if gap > chunk_time_interval_minutes
        2. Size: Split if chunk size > chunk_max_size_chars OR count > chunk_max_count
        3. Semantic: Split if cosine similarity < chunk_similarity_threshold

        Args:
            doc: A DocumentUnit representing the document the chunks belong to.
            msgs: A list of dictionaries representing the messages to process.

        Returns:
            A list of ChunkRecord objects.
        """
        if not msgs:
            return []

        from sentence_transformers.util import cos_sim

        # Sort messages by timestamp
        sorted_msgs = sorted(msgs, key=lambda mm: float(mm.get("ts", 0.0)))

        # Pre-calculate embeddings for all messages
        texts_for_embedding = [m.get("text", "") or " " for m in sorted_msgs]
        model = self._get_embedding_model()
        embeddings = model.encode(texts_for_embedding)

        chunks: List[ChunkRecord] = []

        current_chunk_msgs: List[dict] = []
        current_chunk_size = 0
        current_chunk_start_idx = 0

        prev_msg_ts: Optional[float] = None
        prev_msg_emb: Optional[Any] = None

        for idx, (m, emb) in enumerate(zip(sorted_msgs, embeddings, strict=False)):
            text = self._format_message_for_chunk(m)
            msg_len = len(text)
            ts = float(m.get("ts", 0.0))

            should_split = False

            # 1. Time-based splitting
            if prev_msg_ts is not None:
                delta_minutes = (ts - prev_msg_ts) / 60.0
                if delta_minutes > self.config.chunk_time_interval_minutes:
                    should_split = True

            # 2. Size/Count-based splitting (check against accumulated chunk)
            if not should_split and current_chunk_msgs:
                if (current_chunk_size + msg_len > self.config.chunk_max_size_chars) or (
                    len(current_chunk_msgs) + 1 > self.config.chunk_max_count
                ):
                    should_split = True

            # 3. Semantic splitting
            if not should_split and prev_msg_emb is not None:
                similarity = cos_sim(emb, prev_msg_emb).item()
                if similarity < self.config.chunk_similarity_threshold:
                    should_split = True

            if should_split and current_chunk_msgs:
                chunks.append(self._finalize_chunk(doc, current_chunk_start_idx, current_chunk_msgs))

                current_chunk_msgs = []
                current_chunk_size = 0
                current_chunk_start_idx = idx

            current_chunk_msgs.append(m)
            current_chunk_size += msg_len
            prev_msg_ts = ts
            prev_msg_emb = emb

        if current_chunk_msgs:
            chunks.append(self._finalize_chunk(doc, current_chunk_start_idx, current_chunk_msgs))

        return chunks

    def __process_channel_messages(
        self, channel: dict, messages: List[dict]
    ) -> Tuple[List[DocumentUnit], List[ChunkRecord]]:
        """Convert raw messages for a channel into DocumentUnit and ChunkRecord objects.

        Args:
            channel: A dictionary representing a Slack channel.
            messages: A list of dictionaries representing the messages to process.

        Returns:
            A tuple containing a list of DocumentUnit objects and a list of ChunkRecord objects.
        """
        docs_for_channel: List[DocumentUnit] = []
        chunks_for_channel: List[ChunkRecord] = []
        channel_threads = self._extract_threads(channel, messages)
        for thread_messages in channel_threads.values():
            docs_and_chunks = self._extract_document_from_thread(channel, thread_messages)
            for doc, channel_chunks in docs_and_chunks:
                docs_for_channel.append(doc)
                chunks_for_channel.extend(channel_chunks)
        return docs_for_channel, chunks_for_channel

    def __persist_api_messages(
        self,
        channel_id: str,
        api_messages: List[dict],
        base_dir: Path,
        checkpoint: SlackCheckpoint,
        update_checkpoint: bool,
    ) -> None:
        """Persist freshly fetched API messages and advance checkpoint if requested.

        Args:
            channel_id: The ID of the channel to persist the messages for.
            api_messages: A list of dictionaries representing the API messages to persist.
            base_dir: The base directory to persist the messages to.
            checkpoint: The checkpoint to update.
            update_checkpoint: Whether to update the checkpoint.

        Returns:
            None
        """
        if not api_messages:
            return

        ts_values: List[float] = []
        for msg in api_messages:
            try:
                if msg.get("ts") is not None:
                    ts_values.append(float(msg["ts"]))
            except Exception:
                continue

        if not ts_values:
            return

        ts_start = min(ts_values)
        ts_end = max(ts_values)
        ts_start_dt = datetime.fromtimestamp(ts_start, tz=timezone.utc)
        year = f"{ts_start_dt.year:04d}"
        month = f"{ts_start_dt.month:02d}"
        channel_dir = base_dir / channel_id
        output_dir = channel_dir / year / month
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{channel_id}_{int(ts_start)}_{int(ts_end)}.jsonl.gz"
        try:
            with gzip.open(out_path, "at", encoding="utf-8") as handle:
                for msg in api_messages:
                    handle.write(json.dumps(msg, separators=(",", ":"), ensure_ascii=False))
                    handle.write("\n")
            logger.info(f"Persisted {len(api_messages)} messages to {out_path}")
        except Exception as io_err:
            logger.error(f"Failed to persist messages for channel {channel_id} to {out_path}: {io_err}")
            return

        if update_checkpoint:
            checkpoint.update_channel_ts(channel_id, newest_ts=ts_end, oldest_ts=ts_start)
            try:
                checkpoint.save(base_dir / "checkpoint.json")
                logger.debug(f"Checkpoint persisted for channel {channel_id}")
            except Exception as save_err:
                logger.error(f"Failed to persist checkpoint after channel {channel_id}: {save_err}")

    def __load_cached_messages_for_channel(
        self,
        channel_dir: Path,
        channel_id: str,
        min_ts: Optional[float] = None,
        max_ts: Optional[float] = None,
    ) -> Tuple[List[dict], Optional[float]]:
        """Load cached JSONL.gz files within the provided timestamp window.

        Args:
            channel_dir: The directory to load the cached messages from.
            channel_id: The ID of the channel to load the cached messages for.
            min_ts: The minimum timestamp to load the cached messages from.
            max_ts: The maximum timestamp to load the cached messages from.

        Returns:
            A tuple containing a list of dictionaries representing the cached messages and
            the maximum timestamp seen in the cached messages.
        """
        if not channel_dir.exists():
            return [], None

        lower_bound = min_ts if min_ts is not None else float("-inf")
        upper_bound = max_ts if max_ts is not None else float("inf")

        pattern = re.compile(rf"{re.escape(channel_id)}_(?P<start>\d+)_(?P<end>\d+)\.jsonl\.gz$")
        candidates: List[Tuple[float, float, Path]] = []
        for gz_file in channel_dir.rglob("*.jsonl.gz"):
            match = pattern.match(gz_file.name)
            if not match:
                continue
            start_ts = float(match.group("start"))
            end_ts = float(match.group("end"))
            if end_ts <= lower_bound or start_ts >= upper_bound:
                continue
            candidates.append((start_ts, end_ts, gz_file))

        if not candidates:
            return [], None

        candidates.sort(key=lambda item: item[0])
        loaded_messages: List[dict] = []
        max_seen_ts: Optional[float] = None

        for start_ts, end_ts, gz_path in candidates:
            try:
                with gzip.open(gz_path, "rt", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            message = json.loads(line)
                        except json.JSONDecodeError as decode_err:
                            logger.warning(f"Skipping corrupted JSON line in {gz_path}: {decode_err}")
                            continue
                        ts_val = message.get("ts")
                        try:
                            ts_float = float(ts_val)
                        except (TypeError, ValueError):
                            continue
                        if ts_float < lower_bound or ts_float > upper_bound:
                            continue
                        loaded_messages.append(message)
                        if max_seen_ts is None or ts_float > max_seen_ts:
                            max_seen_ts = ts_float
                logger.debug(
                    f"Loaded cached file {gz_path} covering {start_ts}..{end_ts} (messages={len(loaded_messages)})"
                )
            except Exception as read_err:
                logger.error(f"Failed to read cached file {gz_path}: {read_err}")

        return loaded_messages, max_seen_ts

    # --------- Document/thread building helpers ----------
    def __format_p_ts(self, ts_str: str) -> str:
        """Format a timestamp string for a Slack permalink.

        Args:
            ts_str: A string representing a Slack timestamp.

        Returns:
            A string representing the formatted timestamp for a Slack permalink.
        """
        secs, frac = (str(ts_str).split(".") + ["0"])[:2]
        frac6 = (frac + "000000")[:6]
        return f"p{secs}{frac6}"

    def __first_ts(self, msgs: List[dict]) -> Optional[str]:
        """Get the first timestamp from a list of messages.

        Args:
            msgs: A list of dictionaries representing Slack messages.

        Returns:
            A string representing the first timestamp from the list of messages.
        """
        for m in sorted(msgs, key=lambda x: float(x.get("ts", 0.0))):
            if "ts" in m:
                return str(m["ts"])
        return None

    def __make_doc(
        self,
        channel: dict,
        source_doc_id: str,
        title: str,
        msgs: List[dict],
        tags: List[str],
        meta: dict,
    ) -> DocumentUnit:
        """Make a DocumentUnit from a list of messages.

        Args:
            channel: A dictionary representing a Slack channel.
            source_doc_id: A string representing the source document id.
            title: A string representing the title of the document.
            msgs: A list of dictionaries representing Slack messages.
            tags: A list of strings representing the tags of the document.
            meta: A dictionary representing the metadata of the document.

        Returns:
            A DocumentUnit representing the document.
        """
        texts = [m.get("text", "") for m in msgs]
        content_hash = DocumentUnit.compute_hash("\n".join(texts).encode("utf-8")) if texts else None
        created = min((float(m["ts"]) for m in msgs if "ts" in m), default=None)
        updated = max((float(m["ts"]) for m in msgs if "ts" in m), default=None)
        created_dt = datetime.fromtimestamp(created, tz=timezone.utc) if created else None
        updated_dt = datetime.fromtimestamp(updated, tz=timezone.utc) if updated else None
        domain = self.config.workspace_domain
        first_ts = self.__first_ts(msgs)
        uri = (
            f"https://{domain}/archives/{channel['id']}/{self.__format_p_ts(first_ts)}"
            if (domain and first_ts)
            else None
        )

        # Author resolution from first message
        author_name = None
        if msgs:
            user_id = msgs[0].get("user")
            if user_id:
                author_name = self._resolve_user_display(user_id)

        # Parent ID logic
        parent_id = None
        if meta.get("is_thread"):
            parent_id = channel.get("id")

        return DocumentUnit(
            document_id=f"slack|{self.source_id}|{source_doc_id}",
            source="slack",
            source_instance_id=self.source_id,
            source_doc_id=source_doc_id,
            uri=uri,
            title=title,
            author=author_name,
            parent_id=parent_id,
            language=None,
            source_created_at=created_dt,
            source_updated_at=updated_dt,
            system_tags=tags,
            source_metadata=meta,
            content_hash=content_hash,
        )

    def __process_thread_messages(
        self, channel: dict, ch_name: str, msgs: List[dict], thread_ts: str
    ) -> List[Tuple[DocumentUnit, List[ChunkRecord]]]:
        """Process a list of messages for a thread.

        Args:
            channel: A dictionary representing a Slack channel.
            ch_name: A string representing the name of the channel.
            msgs: A list of dictionaries representing Slack messages.
            thread_ts: A string representing the timestamp of the thread.

        Returns:
            A list of tuples, each containing a DocumentUnit and a list of ChunkRecords.
        """
        if not msgs:
            return []

        title_time_dt = datetime.fromtimestamp(float(thread_ts), tz=timezone.utc)
        source_doc_id = f"Thread:{channel['id']}:{int(title_time_dt.timestamp())}"
        meta = {
            "is_thread": True,
            "channel_id": channel["id"],
            "channel_name": ch_name,
            "thread_ts": thread_ts,
            "message_count": len(msgs),
        }
        doc = self.__make_doc(channel, str(thread_ts), source_doc_id, msgs, ["slack", "thread"], meta)
        chunks: List[ChunkRecord] = self._build_chunks_for_messages(doc, msgs)
        return [(doc, chunks)]

    def __process_windowed_messages(
        self, channel: dict, ch_name: str, msgs: List[dict], window_minutes: int
    ) -> List[Tuple[DocumentUnit, List[ChunkRecord]]]:
        """Process a list of messages for a window.

        Args:
            channel: A dictionary representing a Slack channel.
            ch_name: A string representing the name of the channel.
            msgs: A list of dictionaries representing Slack messages.
            window_minutes: An integer representing the number of minutes in the window.

        Returns:
            A list of tuples, each containing a DocumentUnit and a list of ChunkRecords.
        """
        results: List[Tuple[DocumentUnit, List[ChunkRecord]]] = []
        if not msgs:
            return results
        # Window used to group messages into documents
        window_delta = timedelta(minutes=window_minutes)
        # Get the start timestamp of the window
        win_start = datetime.fromtimestamp(float(msgs[0]["ts"]), tz=timezone.utc)
        bucket: List[dict] = []
        for m in msgs:
            ts_dt = datetime.fromtimestamp(float(m.get("ts", 0.0)), tz=timezone.utc)
            # If the message timestamp is outside the window, start a new document
            if ts_dt >= win_start + window_delta and bucket:
                win_end = ts_dt
                source_doc_id = f"Channel:{channel['id']}:{int(win_start.timestamp())}"
                title = f"Channel {ch_name} {win_start.isoformat()} - {win_end.isoformat()}"
                meta = {
                    "is_thread": False,
                    "channel_id": channel["id"],
                    "channel_name": ch_name,
                    "window_start_ts": win_start.timestamp(),
                    "window_end_ts": win_end.timestamp(),
                    "message_count": len(bucket),
                }
                # Create DocumentUnit and build Chunks
                # Copy the bucket to avoid modifying the original list.
                # Bucket is a reference that is emptied after the document is created
                bucket_copy = list(bucket)
                doc = self.__make_doc(
                    channel,
                    source_doc_id,
                    title,
                    bucket_copy,
                    ["slack", "channel"],
                    meta,
                )
                chunks: List[ChunkRecord] = self._build_chunks_for_messages(doc, bucket_copy)
                results.append((doc, chunks))
                bucket = []
                win_start = ts_dt
            bucket.append(m)
        # If there are remaining messages, create a final document
        if bucket:
            last_ts = float(bucket[-1].get("ts", win_start.timestamp()))
            win_end = datetime.fromtimestamp(last_ts, tz=timezone.utc)
            source_doc_id = f"Channel:{channel['id']}:{int(win_start.timestamp())}"
            title = f"Channel {ch_name} {win_start.isoformat()} - {win_end.isoformat()}"
            meta = {
                "is_thread": False,
                "channel_id": channel["id"],
                "channel_name": ch_name,
                "window_start_ts": win_start.timestamp(),
                "window_end_ts": win_end.timestamp(),
                "message_count": len(bucket),
            }
            bucket_copy = list(bucket)
            doc = self.__make_doc(channel, source_doc_id, title, bucket_copy, ["slack", "channel"], meta)
            chunks = self._build_chunks_for_messages(doc, bucket_copy)
            results.append((doc, chunks))
        return results

    def collect_documents_and_chunks(
        self,
        checkpoint: Checkpoint,
        update_checkpoint: bool = True,
        use_cached_data: bool = True,
        filters: Optional[Dict[str, List[str]]] = None,
    ) -> Tuple[List[DocumentUnit], List[ChunkRecord]]:
        """Collect Slack documents (threads/windows) and build flat chunks in one pass.

        Behavior:
            - Enumerates conversations and messages since the provided `checkpoint`.
            - Optionally rehydrates previously downloaded JSONL.gz cache before calling Slack APIs.
            - Builds `DocumentUnit` objects for:
              - Threads (is_thread=True, identified by `thread_ts`)
              - Channel windows (is_thread=False) bounded by `window_start_ts`/`window_end_ts`
            - Builds flat `ChunkRecord` objects (level=0, parent_chunk_id=None), one per
              message belonging to each document. Chunk text is a Slack-aware, LLM-friendly
              line:
                  "[YYYY-MM-DD HH:MM UTC] Display Name: normalized text"
              where:
                - Slack mentions <@UXXXX> are replaced with @DisplayName using the cached
                  `users.info` results.
                - Simple tags (URLs, email addresses) are extracted via regex and added
                  to `ChunkRecord.system_tags` for filtering/faceting.
            - Honors rate limiting using the adaptive limiter; also records Prometheus
              metrics for API calls and operation latency/items.
            - Updates `checkpoint.oldest_ts` to the newest seen message timestamp at the end.

        Args:
            checkpoint: Per-channel incremental marker.
            update_checkpoint: Whether to persist checkpoint state after writing new data.
            use_cached_data: If True, replay cached JSONL.gz files newer than the checkpoint
                before issuing API calls.
            filters: Optional list of identifiers by filtering type to restrict processing.

        Returns:
            Tuple[List[DocumentUnit], List[ChunkRecord]]: The documents discovered and
            their corresponding flat chunks.

        Todo: Group by space like Jira? "data/confluence/{source_id}/{space_key}/{space_key}.jsonl.gz"

        Notes:
            - Documents and chunks are returned but not persisted here; the caller is
              responsible for storage and embedding workflows.
            - Future hierarchical chunking can assign `parent_chunk_id` and `level>0`.
        """
        start_time = perf_counter()
        try:
            if not isinstance(checkpoint, SlackCheckpoint):
                raise ValueError("Invalid checkpoint type, expected SlackCheckpoint")

            all_documents: List[DocumentUnit] = []
            all_chunks: List[ChunkRecord] = []
            # Ensure base storage directory exists: <storage_path>/<source_id>
            base_dir = Path(self.storage_path) / self.source_id
            base_dir.mkdir(parents=True, exist_ok=True)

            if filters and "channel_ids" in filters:
                channels = [
                    self._get_channel_metadata(channel_id, use_cached_data) for channel_id in filters["channel_ids"]
                ]
            else:
                channels = self._list_conversations(self.config.channel_types)
            logger.info(f"Found {len(channels)} channels")
            for channel in channels:
                channel_id = channel["id"]
                channel_dir = base_dir / channel_id
                base_latest = checkpoint.get_latest_ts(channel_id)
                base_earliest = checkpoint.get_earliest_ts(channel_id)
                cached_messages, cached_max_ts = (
                    self.__load_cached_messages_for_channel(
                        channel_dir,
                        channel_id,
                        max_ts=base_latest,
                        min_ts=base_earliest,
                    )
                    if use_cached_data
                    else ([], None)
                )
                logger.debug(
                    f"Loaded {len(cached_messages)} cached messages for channel {channel_id} since {base_earliest}"
                )
                messages: List[dict] = list(cached_messages)

                api_earliest: Optional[float] = None
                if use_cached_data:
                    api_earliest = base_earliest
                    if cached_max_ts is not None:
                        if api_earliest is None:
                            api_earliest = cached_max_ts
                        else:
                            api_earliest = max(api_earliest, cached_max_ts)

                try:
                    api_messages = self._list_messages(channel_id, checkpoint, latest_ts_override=api_earliest)
                    # The first messages in api_messages may be duplicates of the last messages in cached_messages
                    # so we need to remove them.
                    if len(cached_messages) > 0:
                        api_messages = [msg for msg in api_messages if msg["ts"] > cached_messages[-1]["ts"]]
                except Exception as e:
                    logger.error(
                        f"Error listing messages for channel (messages list process interrupted, not all messages collected): {channel_id}: {e}"
                    )
                    api_messages = []

                messages.extend(api_messages)
                logger.info(
                    f"Channel {channel_id}: cache_messages={len(cached_messages)} "
                    f"api_messages={len(api_messages)} since {api_earliest}"
                )
                self.__persist_api_messages(
                    channel_id=channel_id,
                    api_messages=api_messages,
                    base_dir=base_dir,
                    checkpoint=checkpoint,
                    update_checkpoint=update_checkpoint,
                )

                if len(messages) > 0:
                    self._persist_channel_metadata(channel_dir, channel)

                docs_for_channel, chunks_for_channel = self.__process_channel_messages(channel, messages)
                all_documents.extend(docs_for_channel)
                all_chunks.extend(chunks_for_channel)
                logger.info(
                    f"documents built for channel {channel_id}: channel_docs={len(docs_for_channel)} "
                    f"channel_chunks={len(chunks_for_channel)} "
                    f"totals={{'docs': {len(all_documents)}, 'chunks': {len(all_chunks)}}}"
                )

            if update_checkpoint:
                try:
                    checkpoint.save(base_dir / "checkpoint.json")
                except Exception as e:
                    logger.error(f"Failed to write checkpoint at end of run: {e}")

            return all_documents, all_chunks
        finally:
            duration = perf_counter() - start_time
            OP_LATENCY.labels(
                source="slack", source_id=self.source_id, operation="collect_documents_and_chunks"
            ).observe(duration)
            logger.debug(f"collect_documents_and_chunks took {duration:.4f} seconds")

    def collect_cached_documents_and_chunks(
        self,
        filters: Optional[Dict[str, List[str]]] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> Tuple[List[DocumentUnit], List[ChunkRecord]]:
        """Rehydrate documents and chunks exclusively from cached JSONL data.

        Args:
            filters: Optional list of identifiers by filtering type to restrict processing.
            date_from: Optional UTC datetime lower bound (inclusive).
            date_to: Optional UTC datetime upper bound (inclusive).

        Returns:
            Tuple[List[DocumentUnit], List[ChunkRecord]]: The documents discovered and
            their corresponding flat chunks.
        """
        base_dir = Path(self.storage_path) / self.source_id
        if not base_dir.exists():
            logger.warning("Cache directory does not exist; returning empty result")
            return [], []

        all_documents: List[DocumentUnit] = []
        all_chunks: List[ChunkRecord] = []

        if filters and "channel_ids" in filters:
            # Sort and deduplicate channel IDs
            target_channels = sorted(set(filters["channel_ids"]))
        else:
            target_channels = [entry.name for entry in base_dir.iterdir() if entry.is_dir()]

        min_ts = date_from.timestamp() if date_from else None
        max_ts = date_to.timestamp() if date_to else None

        logger.info(
            f"Collecting cached documents for channels={target_channels if filters and 'channel_ids' in filters else 'ALL'} "
            f"range=({date_from}, {date_to})"
        )

        for channel_id in target_channels:
            channel_dir = base_dir / channel_id
            channel_meta = self._load_channel_metadata(channel_dir, channel_id)
            messages, _ = self.__load_cached_messages_for_channel(channel_dir, channel_id, min_ts=min_ts, max_ts=max_ts)
            if not messages:
                logger.debug(f"No cached messages found for channel {channel_id} in requested range")
                continue
            docs, chunks = self.__process_channel_messages(channel_meta, messages)
            all_documents.extend(docs)
            all_chunks.extend(chunks)
            logger.info(
                f"Loaded {len(messages)} cached messages for channel {channel_id} -> "
                f"{len(docs)} docs / {len(chunks)} chunks"
            )

        logger.info(f"Cache-only replay completed: documents={len(all_documents)} chunks={len(all_chunks)}")
        return all_documents, all_chunks


log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)
logging.basicConfig(level=log_level)
logger = logging.getLogger(__name__)
logger.setLevel(log_level)

if __name__ == "__main__":
    # Enable console logging at DEBUG level
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    has_stream = False
    for h in root_logger.handlers:
        if isinstance(h, logging.StreamHandler):
            h.setLevel(logging.DEBUG)
            has_stream = True
    if not has_stream:
        sh = logging.StreamHandler()
        sh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
        sh.setFormatter(formatter)
        root_logger.addHandler(sh)

    slack_config = SlackConfig(
        id="slack-main",
        channel_types=["private_channel", "public_channel", "im", "mpim"],
        initial_lookback_days=180,
        channel_window_minutes=60,
        workspace_domain=os.getenv("SLACK_WORKSPACE_DOMAIN", "https://slack.com"),
        user_cache_path=os.path.join("data", "slack", "slack-main", "slack_users_cache.json"),
        user_cache_ttl_seconds=0,
    )
    entity_manager = EntityManager(storage_path="data/entities.json")
    slack_source = SlackSource(
        config=slack_config,
        secrets={"bot_token": os.getenv("SLACK_BOT_TOKEN")},
        storage_path=os.path.join("data", "slack"),
        entity_manager=entity_manager,
    )
    checkpoint_path = Path("data") / "slack" / "slack-main" / "checkpoint.json"
    if checkpoint_path.exists():
        slack_checkpoint = SlackCheckpoint.load(checkpoint_path, slack_source.config)
    else:
        slack_checkpoint = SlackCheckpoint(config=slack_source.config)
    docs, chunks = slack_source.collect_documents_and_chunks(slack_checkpoint, use_cached_data=True)
    # docs, chunks = slack_source.collect_cached_documents_and_chunks(
    #     date_from=datetime(2025, 11, 13, 0, 0, 0, tzinfo=timezone.utc),
    #     date_to=datetime(2025, 11, 17, 0, 0, 0, tzinfo=timezone.utc),
    # )
    logger.info(f"Collected documents={len(docs)} chunks={len(chunks)}")
