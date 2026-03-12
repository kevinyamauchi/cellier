"""CPU-side chunk cache with async loading, priority queue, and cancellation."""

from __future__ import annotations

import heapq
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING, Callable

from psygnal import Signal

from cellier.types import ChunkData, ChunkRequest

if TYPE_CHECKING:
    from concurrent.futures import Future

    import numpy as np

    from cellier.slicer.slicer import AsynchronousDataSlicer


class ChunkManager:
    """CPU-side chunk cache with async loading, priority queue, and cancellation.

    Maintains an LRU cache of loaded chunk numpy arrays.  On a cache miss the
    chunk is queued for background loading via the shared
    :class:`~cellier.slicer.slicer.AsynchronousDataSlicer` thread pool.
    Requests are dispatched in ascending priority order (lower value = higher
    priority).  All pending requests belonging to a visual can be silently
    cancelled with :meth:`cancel_visual`.

    When a background load completes successfully the :attr:`chunk_loaded`
    signal is emitted with the resulting :class:`~cellier.types.ChunkData`.

    Parameters
    ----------
    slicer : AsynchronousDataSlicer
        The application-wide async slicer whose thread pool is used for all
        background I/O.  ``ChunkManager`` submits directly to
        ``slicer._thread_pool`` so that chunk loading shares the same worker
        pool as all other data requests.
    loader : Callable[[ChunkRequest], np.ndarray]
        Synchronous callable that reads one chunk's voxel data from storage.
        Called from a background thread; must be thread-safe.
    max_cache_bytes : int
        Upper bound on total cached voxel data in bytes.  Oldest entries (by
        insertion / last-access time) are evicted when this limit is exceeded.
    """

    chunk_loaded: Signal = Signal(ChunkData)

    def __init__(
        self,
        slicer: AsynchronousDataSlicer,
        loader: Callable[[ChunkRequest], np.ndarray],
        max_cache_bytes: int,
    ) -> None:
        self._thread_pool = slicer._thread_pool
        self._loader = loader
        self._max_cache_bytes = max_cache_bytes

        # LRU cache: (scale_index, chunk_index) → ndarray, ordered oldest→newest.
        self._cache: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()
        self._cache_bytes: int = 0

        # Lock guards _cache, _cache_bytes, _pending, and _cancelled.
        self._lock = threading.Lock()

        # _pending maps visual_id → set of (scale_index, chunk_index) currently
        # in-flight or queued.
        self._pending: dict[str, set[tuple[int, int]]] = {}

        # _cancelled holds (visual_id, scale_index, chunk_index) tuples whose
        # results should be silently discarded.
        self._cancelled: set[tuple[str, int, int]] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def request_chunks(
        self,
        requests: list[ChunkRequest],
        visual_id: str,
    ) -> tuple[list[ChunkData], int]:
        """Return cached chunks and schedule missing ones for async load.

        Parameters
        ----------
        requests : list[ChunkRequest]
            Chunks to retrieve, each carrying a ``priority`` value.
        visual_id : str
            Visual identifier used to group in-flight requests for
            cancellation via :meth:`cancel_visual`.

        Returns
        -------
        available : list[ChunkData]
            Chunks already in the CPU cache (returned immediately, with no
            background I/O).
        pending_count : int
            Number of chunks that were not cached and have been submitted for
            background loading.
        """
        available: list[ChunkData] = []
        heap: list[tuple[float, ChunkRequest]] = []

        with self._lock:
            for req in requests:
                key = (req.scale_index, req.chunk_index)
                if key in self._cache:
                    # Cache hit — promote to most-recently-used position.
                    self._cache.move_to_end(key)
                    available.append(
                        ChunkData(
                            chunk_index=req.chunk_index,
                            scale_index=req.scale_index,
                            data=self._cache[key],
                        )
                    )
                else:
                    heapq.heappush(heap, (req.priority, req))

            # Register all cache-miss keys as in-flight for this visual.
            in_flight_keys = {(r.scale_index, r.chunk_index) for _, r in heap}
            if in_flight_keys:
                if visual_id not in self._pending:
                    self._pending[visual_id] = set()
                self._pending[visual_id].update(in_flight_keys)

        # Dispatch to thread pool in ascending priority order.
        pending_count = 0
        while heap:
            _, req = heapq.heappop(heap)
            future: Future[np.ndarray] = self._thread_pool.submit(self._loader, req)
            future.add_done_callback(lambda f, r=req: self._on_chunk_loaded(f, r))
            pending_count += 1

        return available, pending_count

    def cancel_visual(self, visual_id: str) -> None:
        """Discard any pending loads for *visual_id*.

        Future completions for the cancelled requests will be silently dropped
        without emitting :attr:`chunk_loaded`.

        Parameters
        ----------
        visual_id : str
            Visual whose pending requests should be cancelled.
        """
        with self._lock:
            if visual_id in self._pending:
                for scale_idx, chunk_idx in self._pending.pop(visual_id):
                    self._cancelled.add((visual_id, scale_idx, chunk_idx))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _on_chunk_loaded(self, future: Future[np.ndarray], req: ChunkRequest) -> None:
        """Background-thread callback invoked when a load future completes."""
        if future.cancelled():
            return

        try:
            data = future.result()
        except Exception:
            return

        cancel_key = (req.visual_id, req.scale_index, req.chunk_index)
        cache_key = (req.scale_index, req.chunk_index)

        with self._lock:
            # Drop silently if the visual was cancelled.
            if cancel_key in self._cancelled:
                self._cancelled.discard(cancel_key)
                return

            # Remove from the in-flight tracking set.
            vis_pending = self._pending.get(req.visual_id)
            if vis_pending is not None:
                vis_pending.discard((req.scale_index, req.chunk_index))
                if not vis_pending:
                    del self._pending[req.visual_id]

            # Insert into cache and evict if necessary.
            self._cache[cache_key] = data
            self._cache_bytes += data.nbytes
            self._evict_if_needed()

        self.chunk_loaded.emit(
            ChunkData(
                chunk_index=req.chunk_index,
                scale_index=req.scale_index,
                data=data,
            )
        )

    def _evict_if_needed(self) -> None:
        """Evict LRU entries until total cached bytes ≤ max_cache_bytes.

        Must be called while holding ``self._lock``.
        """
        while self._cache_bytes > self._max_cache_bytes and self._cache:
            _key, evicted = self._cache.popitem(last=False)
            self._cache_bytes -= evicted.nbytes
