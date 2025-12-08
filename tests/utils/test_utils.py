import os
import json
import time
from src.utils.persistent_cache import PersistentCache

def test_persistent_cache_get_and_set(temp_dir):
    cache_path = os.path.join(temp_dir, "cache.json")
    cache = PersistentCache(cache_path, ttl_seconds=10)

    cache.set("key", {"data": "value"})

    retrieved_value = cache.get("key")
    assert retrieved_value is not None
    assert retrieved_value["data"] == "value"

def test_persistent_cache_non_existent_key(temp_dir):
    cache_path = os.path.join(temp_dir, "cache.json")
    cache = PersistentCache(cache_path)

    retrieved_value = cache.get("non_existent_key", default="default_value")
    assert retrieved_value == "default_value"

def test_persistent_cache_expired_entry(temp_dir):
    cache_path = os.path.join(temp_dir, "cache.json")
    cache = PersistentCache(cache_path, ttl_seconds=1)

    cache.set("key", {"data": "value"})
    time.sleep(2)

    retrieved_value = cache.get("key", default="default_value")
    assert retrieved_value == "default_value"
