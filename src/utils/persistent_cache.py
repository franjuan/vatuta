"""Persistent cache implementation with TTL support."""

import json
import logging
import os
import tempfile
import threading
import time
from typing import Any, Dict

logger = logging.getLogger(__name__)


class PersistentCache:
    """Simple JSON-backed persistent cache with TTL per entry.

    Each entry is expected to include a 'cached_at' unix timestamp (seconds).
    If absent, it's set automatically on set().
    """

    def __init__(self, path: str, ttl_seconds: int = 7 * 24 * 60 * 60):
        """Initialize the persistent cache.

        Args:
            path: Path to the cache file.
            ttl_seconds: Time-to-live for cache entries in seconds. 0 means no TTL.
        """
        self._path = path
        self._ttl = int(ttl_seconds)
        self._lock = threading.Lock()
        dir_name = os.path.dirname(self._path) or "."
        os.makedirs(dir_name, exist_ok=True)
        self._data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        try:
            if not os.path.exists(self._path):
                return {}
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception as e:
            logger.warning(f"Failed to load cache from {self._path}: {e}")
            return {}

    def _save(self) -> None:
        dir_name = os.path.dirname(self._path) or "."
        fd, tmp_path = tempfile.mkstemp(prefix=".tmp_cache_", dir=dir_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tmpf:
                json.dump(self._data, tmpf, ensure_ascii=False)
            os.replace(tmp_path, self._path)
        except Exception as e:
            logger.warning(f"Failed to save cache to {self._path}: {e}")
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def _is_valid(self, entry: Dict[str, Any]) -> bool:
        try:
            cached_at = int(entry.get("cached_at", 0))
        except Exception:
            return False
        return self._ttl == 0 or cached_at + self._ttl > int(time.time())

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the cache.

        Args:
            key: Cache key.
            default: Default value if key not found or expired.

        Returns:
            Cached value or default.
        """
        with self._lock:
            entry = self._data.get(key)
            if entry and self._is_valid(entry):
                return entry
            return default

    def set(self, key: str, value: Dict[str, Any]) -> None:
        """Set a value in the cache.

        Args:
            key: Cache key.
            value: Value to cache.
        """
        with self._lock:
            if "cached_at" not in value:
                value = dict(value)
                value["cached_at"] = int(time.time())
            self._data[key] = value
            self._save()
