import concurrent.futures
import numpy as np
import pytest
from zarr_proxy.cache import ThreadSafeLRUCache, get_bytes_size, get_numpy_size


def test_cache_byte_bounds_eviction():
    # Cache limited to 1000 bytes
    cache = ThreadSafeLRUCache(maxsize_bytes=1000, getsizeof=get_bytes_size)

    # Put 3 items of 400 bytes each
    b1 = b"x" * 400
    b2 = b"y" * 400
    b3 = b"z" * 400

    cache.put("k1", b1)
    cache.put("k2", b2)
    assert cache.currsize_bytes == 800
    assert "k1" in cache
    assert "k2" in cache

    # Putting k3 (400 bytes) pushes total to 1200 > 1000, so k1 must be evicted!
    cache.put("k3", b3)
    assert cache.currsize_bytes == 800
    assert "k1" not in cache
    assert "k2" in cache
    assert "k3" in cache


def test_cache_rejects_oversized_item():
    cache = ThreadSafeLRUCache(maxsize_bytes=500, getsizeof=get_bytes_size)
    big = b"a" * 600
    accepted = cache.put("big", big)
    assert not accepted
    assert "big" not in cache
    assert cache.currsize_bytes == 0


def test_cache_numpy_size_tracking():
    cache = ThreadSafeLRUCache(maxsize_bytes=10000, getsizeof=get_numpy_size)
    arr = np.zeros((10, 10), dtype=np.uint8)  # 100 bytes
    cache.put("arr1", arr)
    assert cache.currsize_bytes == 100

    arr2 = np.zeros((10, 10), dtype=np.float32)  # 400 bytes
    cache.put("arr2", arr2)
    assert cache.currsize_bytes == 500


def test_cache_thread_safety():
    cache = ThreadSafeLRUCache(maxsize_bytes=50000, getsizeof=get_bytes_size)

    def worker(worker_id):
        for i in range(100):
            key = f"w_{worker_id}_{i % 10}"
            val = b"1" * 50
            cache.put(key, val)
            _ = cache.get(key)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker, w) for w in range(8)]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    assert cache.currsize_bytes <= 50000
    stats = cache.stats()
    assert stats["hits"] > 0
    assert stats["items_count"] > 0
