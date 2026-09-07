"""
Thread-safe, bounded LRU caches with exact byte-level memory tracking.
"""

from __future__ import annotations

import os
import sys
import threading
from typing import Any, Callable, Dict, Optional, Tuple
from cachetools import LRUCache
import numpy as np

# Default limits: 2GB for raw chunks, 1GB for processed outputs
DEFAULT_RAW_CACHE_BYTES = int(os.environ.get("RAW_CACHE_BYTES", 2 * 1024 * 1024 * 1024))
DEFAULT_PROCESSED_CACHE_BYTES = int(os.environ.get("PROCESSED_CACHE_BYTES", 1 * 1024 * 1024 * 1024))


def get_numpy_size(value: Any) -> int:
    """Accurately compute memory size in bytes for cached objects."""
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)
    return sys.getsizeof(value)


def get_bytes_size(value: Any) -> int:
    """Compute size for raw compressed byte buffers."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)
    return sys.getsizeof(value)


class ThreadSafeLRUCache:
    """
    Thread-safe wrapper around cachetools.LRUCache that enforces
    strict memory limits measured in bytes.
    """

    def __init__(self, maxsize_bytes: int, getsizeof: Optional[Callable[[Any], int]] = None, name: str = "cache"):
        self.name = name
        self.maxsize_bytes = maxsize_bytes
        self.getsizeof_fn = getsizeof or get_numpy_size
        self._lock = threading.RLock()
        self._cache = LRUCache(maxsize=maxsize_bytes, getsizeof=self.getsizeof_fn)
        self._hits = 0
        self._misses = 0

    def get(self, key: Any) -> Optional[Any]:
        with self._lock:
            if key in self._cache:
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def put(self, key: Any, value: Any) -> bool:
        """
        Store value in cache. Returns True if stored, False if rejected
        (e.g., if single value exceeds max cache size).
        """
        size = self.getsizeof_fn(value)
        if size > self.maxsize_bytes:
            # Cannot fit even in an empty cache
            return False

        with self._lock:
            try:
                self._cache[key] = value
                return True
            except ValueError:
                # In case LRUCache raises if item > maxsize
                return False

    def __contains__(self, key: Any) -> bool:
        with self._lock:
            return key in self._cache

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    @property
    def currsize_bytes(self) -> int:
        with self._lock:
            return self._cache.currsize

    @property
    def maxsize_bytes_limit(self) -> int:
        return self.maxsize_bytes

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total_requests = self._hits + self._misses
            hit_ratio = (self._hits / total_requests) if total_requests > 0 else 0.0
            return {
                "name": self.name,
                "items_count": len(self._cache),
                "currsize_bytes": self._cache.currsize,
                "currsize_mb": round(self._cache.currsize / (1024 * 1024), 2),
                "maxsize_bytes": self.maxsize_bytes,
                "maxsize_mb": round(self.maxsize_bytes / (1024 * 1024), 2),
                "hits": self._hits,
                "misses": self._misses,
                "hit_ratio": round(hit_ratio, 4),
            }


# Global singleton cache instances
_raw_cache: Optional[ThreadSafeLRUCache] = None
_processed_cache: Optional[ThreadSafeLRUCache] = None
_init_lock = threading.Lock()


def get_raw_chunk_cache(maxsize_bytes: Optional[int] = None) -> ThreadSafeLRUCache:
    """Retrieve the singleton raw chunk cache instance."""
    global _raw_cache
    if _raw_cache is None:
        with _init_lock:
            if _raw_cache is None:
                limit = maxsize_bytes or DEFAULT_RAW_CACHE_BYTES
                _raw_cache = ThreadSafeLRUCache(
                    maxsize_bytes=limit,
                    getsizeof=get_numpy_size,
                    name="raw_chunk_cache"
                )
    return _raw_cache


def get_processed_chunk_cache(maxsize_bytes: Optional[int] = None) -> ThreadSafeLRUCache:
    """Retrieve the singleton processed chunk cache instance."""
    global _processed_cache
    if _processed_cache is None:
        with _init_lock:
            if _processed_cache is None:
                limit = maxsize_bytes or DEFAULT_PROCESSED_CACHE_BYTES
                _processed_cache = ThreadSafeLRUCache(
                    maxsize_bytes=limit,
                    getsizeof=get_bytes_size,
                    name="processed_chunk_cache"
                )
    return _processed_cache
