"""Tests for ChunkManager (Phase C).

Tests are labelled T-C-01 … T-C-06.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import numpy as np

from cellier.types import ChunkData, ChunkRequest
from cellier.utils.chunked_image._chunk_manager import ChunkManager


def make_mock_slicer(max_workers: int = 2) -> MagicMock:
    """Return a mock AsynchronousDataSlicer with a real thread pool attached."""
    slicer = MagicMock()
    slicer._thread_pool = ThreadPoolExecutor(max_workers=max_workers)
    return slicer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CHUNK_SHAPE = (32, 32, 32)
CHUNK_BYTES = int(np.prod(CHUNK_SHAPE)) * np.dtype(np.float32).itemsize  # 131072


def make_request(
    chunk_index: int = 0,
    scale_index: int = 0,
    priority: float = 0.0,
    visual_id: str = "v1",
    scene_id: str = "s1",
) -> ChunkRequest:
    """Build a ChunkRequest with sensible defaults."""
    return ChunkRequest(
        chunk_index=chunk_index,
        scale_index=scale_index,
        priority=priority,
        visual_id=visual_id,
        scene_id=scene_id,
    )


def simple_loader(req: ChunkRequest) -> np.ndarray:
    """Synchronous loader that returns a zeroed float32 chunk."""
    return np.zeros(CHUNK_SHAPE, dtype=np.float32)


# ---------------------------------------------------------------------------
# T-C-01: cache hit returns immediately, loader not called
# ---------------------------------------------------------------------------


def test_cache_hit_returns_immediately():
    """T-C-01: Pre-populated cache → chunk in available; loader not called."""
    call_count = {"n": 0}

    def counting_loader(req: ChunkRequest) -> np.ndarray:
        call_count["n"] += 1
        return simple_loader(req)

    manager = ChunkManager(
        slicer=make_mock_slicer(),
        loader=counting_loader,
        max_cache_bytes=10 * CHUNK_BYTES,
    )

    # Manually insert a chunk into the cache.
    key = (0, 0)
    expected_data = np.ones(CHUNK_SHAPE, dtype=np.float32)
    with manager._lock:
        manager._cache[key] = expected_data
        manager._cache_bytes += expected_data.nbytes

    req = make_request(chunk_index=0, scale_index=0)
    available, pending_count = manager.request_chunks([req], visual_id="v1")

    assert pending_count == 0, "No background load should be triggered on a cache hit"
    assert len(available) == 1
    assert available[0].chunk_index == 0
    assert available[0].scale_index == 0
    np.testing.assert_array_equal(available[0].data, expected_data)
    assert call_count["n"] == 0, "Loader must not be called on a cache hit"


# ---------------------------------------------------------------------------
# T-C-02: cache miss triggers background load; chunk_loaded emitted; chunk cached
# ---------------------------------------------------------------------------


def test_cache_miss_triggers_load():
    """T-C-02: Cache miss → loader called, chunk_loaded emitted, chunk in cache."""
    gate = threading.Event()
    loaded_gate = threading.Event()

    def gated_loader(req: ChunkRequest) -> np.ndarray:
        gate.wait(timeout=5.0)
        return simple_loader(req)

    manager = ChunkManager(
        slicer=make_mock_slicer(), loader=gated_loader, max_cache_bytes=10 * CHUNK_BYTES
    )

    received: list[ChunkData] = []
    manager.chunk_loaded.connect(lambda cd: (received.append(cd), loaded_gate.set()))

    req = make_request(chunk_index=3, scale_index=0)
    available, pending_count = manager.request_chunks([req], visual_id="v1")

    assert len(available) == 0
    assert pending_count == 1

    gate.set()  # release the loader
    assert loaded_gate.wait(timeout=5.0), "chunk_loaded was not emitted in time"

    assert len(received) == 1
    assert received[0].chunk_index == 3
    assert received[0].scale_index == 0

    with manager._lock:
        assert (0, 3) in manager._cache, "Loaded chunk must be stored in the cache"


# ---------------------------------------------------------------------------
# T-C-03: cancellation before load completes suppresses chunk_loaded signal
# ---------------------------------------------------------------------------


def test_cancellation_suppresses_signal():
    """T-C-03: cancel_visual before future completes → chunk_loaded NOT emitted."""
    gate = threading.Event()
    done_event = threading.Event()

    def gated_loader(req: ChunkRequest) -> np.ndarray:
        gate.wait(timeout=5.0)
        done_event.set()
        return simple_loader(req)

    manager = ChunkManager(
        slicer=make_mock_slicer(max_workers=1),
        loader=gated_loader,
        max_cache_bytes=10 * CHUNK_BYTES,
    )

    received: list[ChunkData] = []
    manager.chunk_loaded.connect(received.append)

    req = make_request(chunk_index=7, scale_index=0, visual_id="v1")
    manager.request_chunks([req], visual_id="v1")

    # Cancel before the gate opens.
    manager.cancel_visual("v1")

    # Now let the loader finish.
    gate.set()
    assert done_event.wait(timeout=5.0), "Loader did not complete in time"

    # Give the callback a moment to run, then assert nothing was emitted.
    time.sleep(0.05)
    assert len(received) == 0, "chunk_loaded must not be emitted after cancel_visual"


# ---------------------------------------------------------------------------
# T-C-04: priority ordering — requests dispatched lowest-priority-value first
# ---------------------------------------------------------------------------


def test_priority_ordering():
    """T-C-04: Three requests [3.0, 1.0, 2.0] dispatched in order [1.0, 2.0, 3.0]."""
    order_lock = threading.Lock()
    dispatch_order: list[float] = []
    all_done = threading.Event()

    def recording_loader(req: ChunkRequest) -> np.ndarray:
        with order_lock:
            dispatch_order.append(req.priority)
            if len(dispatch_order) == 3:
                all_done.set()
        return simple_loader(req)

    # Single worker ensures requests are serialised (FIFO queue in thread pool).
    manager = ChunkManager(
        slicer=make_mock_slicer(max_workers=1),
        loader=recording_loader,
        max_cache_bytes=10 * CHUNK_BYTES,
    )

    requests = [
        make_request(chunk_index=0, priority=3.0),
        make_request(chunk_index=1, priority=1.0),
        make_request(chunk_index=2, priority=2.0),
    ]
    manager.request_chunks(requests, visual_id="v1")

    assert all_done.wait(timeout=5.0), "Not all chunks loaded in time"
    assert dispatch_order == [
        1.0,
        2.0,
        3.0,
    ], f"Expected [1.0, 2.0, 3.0], got {dispatch_order}"


# ---------------------------------------------------------------------------
# T-C-05: cache eviction keeps total bytes ≤ max_cache_bytes
# ---------------------------------------------------------------------------


def test_cache_eviction():
    """T-C-05: Fill cache beyond limit; oldest entry evicted; bytes within limit."""
    # Allow exactly 2 chunks worth of bytes.
    max_bytes = 2 * CHUNK_BYTES
    all_done = threading.Event()
    load_count = {"n": 0}
    lock = threading.Lock()

    def counting_loader(req: ChunkRequest) -> np.ndarray:
        data = simple_loader(req)
        with lock:
            load_count["n"] += 1
            if load_count["n"] == 3:
                all_done.set()
        return data

    manager = ChunkManager(
        slicer=make_mock_slicer(max_workers=1),
        loader=counting_loader,
        max_cache_bytes=max_bytes,
    )

    requests = [make_request(chunk_index=i, priority=float(i)) for i in range(3)]
    manager.request_chunks(requests, visual_id="v1")

    assert all_done.wait(timeout=5.0), "Not all chunks loaded in time"
    time.sleep(0.05)  # let the last callback finish

    with manager._lock:
        assert (
            manager._cache_bytes <= max_bytes
        ), f"Cache bytes {manager._cache_bytes} exceeds limit {max_bytes}"
        assert (
            len(manager._cache) == 2
        ), f"Expected 2 cached chunks (oldest evicted), got {len(manager._cache)}"
        # Chunk 0 (oldest inserted) should have been evicted.
        assert (0, 0) not in manager._cache, "Oldest chunk (index 0) should be evicted"


# ---------------------------------------------------------------------------
# T-C-06: cancel_visual only affects the targeted visual; others proceed normally
# ---------------------------------------------------------------------------


def test_multiple_visuals_cancel_independently():
    """T-C-06: Cancelling v1 does not suppress chunk_loaded for v2."""
    gate_v1 = threading.Event()
    gate_v2 = threading.Event()
    v2_done = threading.Event()

    def selective_loader(req: ChunkRequest) -> np.ndarray:
        if req.visual_id == "v1":
            gate_v1.wait(timeout=5.0)
        else:
            gate_v2.wait(timeout=5.0)
        return simple_loader(req)

    manager = ChunkManager(
        slicer=make_mock_slicer(max_workers=2),
        loader=selective_loader,
        max_cache_bytes=10 * CHUNK_BYTES,
    )

    received: list[ChunkData] = []
    manager.chunk_loaded.connect(lambda cd: (received.append(cd), v2_done.set()))

    req_v1 = make_request(chunk_index=0, visual_id="v1", priority=0.0)
    req_v2 = make_request(chunk_index=1, visual_id="v2", priority=0.0)

    manager.request_chunks([req_v1], visual_id="v1")
    manager.request_chunks([req_v2], visual_id="v2")

    # Cancel v1 while both are in flight.
    manager.cancel_visual("v1")

    # Let v2 complete; keep v1 blocked so we can be sure of ordering.
    gate_v2.set()
    assert v2_done.wait(timeout=5.0), "chunk_loaded for v2 was not emitted"

    # Now let v1 finish (it should be silently dropped).
    gate_v1.set()
    time.sleep(0.05)

    chunk_indices = [cd.chunk_index for cd in received]
    assert 1 in chunk_indices, "Chunk from v2 must be emitted"
    assert 0 not in chunk_indices, "Chunk from v1 must be suppressed after cancel"
