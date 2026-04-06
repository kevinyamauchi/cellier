"""Component to handle asynchronous batch loading of chunk data."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any

import numpy as np

from cellier.v2.data.image._image_requests import ChunkRequest
from cellier.v2.logging import _PERF_LOGGER, _SLICER_LOGGER

if TYPE_CHECKING:
    from uuid import UUID

# ---------------------------------------------------------------------------
# PySide6 compatibility patch
#
# QAsyncioTask is a from-scratch reimplementation of asyncio.Task that does
# not inherit from CPython's asyncio.Task.  As a result it is missing several
# CPython-internal methods.  One of these, _make_cancelled_error(), is called
# unconditionally by asyncio.gather's internal _GatheringFuture._done_callback
# when a child task is cancelled.  Without the method the cancellation raises:
#
#   AttributeError: 'QAsyncioTask' object has no attribute '_make_cancelled_error'
#
# The fix is to add the missing method.  The implementation matches CPython's
# exactly (asyncio/tasks.py): construct and return a CancelledError, optionally
# carrying the cancellation message stored on the task.
#
# The hasattr guard makes this a no-op if PySide6 ever adds the method itself.
# ---------------------------------------------------------------------------
try:
    from PySide6.QtAsyncio.events import QAsyncioEventLoop as _QAsyncioEventLoop
    from PySide6.QtAsyncio.tasks import QAsyncioTask as _QAsyncioTask

    # Patch 1: QAsyncioTask._make_cancelled_error
    # Called by asyncio.gather's _GatheringFuture._done_callback when a child
    # task is cancelled.  QAsyncioTask omits this CPython-internal method.
    if not hasattr(_QAsyncioTask, "_make_cancelled_error"):

        def _make_cancelled_error(self):  # type: ignore[misc]
            return asyncio.CancelledError(getattr(self, "_cancel_message", None))

        _QAsyncioTask._make_cancelled_error = _make_cancelled_error  # type: ignore[attr-defined]

    # Patch 2: QAsyncioEventLoop.default_exception_handler
    # When a cancelled task's CancelledError flows through QAsyncioTask._step,
    # the loop's default_exception_handler is called with a context dict that
    # may not contain a 'task' key.  Some PySide6 versions format the message
    # as f"... from task {context['task']._name}" — crashing with KeyError.
    # Additionally, asyncio.gather logs a spurious
    # "_GatheringFuture exception was never retrieved (CancelledError)"
    # message whenever a gather is cancelled — this is expected and harmless.
    # Both cases are handled here.
    _original_exc_handler = _QAsyncioEventLoop.default_exception_handler

    def _safe_default_exception_handler(  # type: ignore[misc]
        self, context: dict
    ) -> None:
        # Silently suppress CancelledError reports — these are always
        # expected when an AsyncSlicer task is cancelled mid-gather and
        # produce no actionable information.
        exc = context.get("exception")
        if isinstance(exc, asyncio.CancelledError):
            return

        task = context.get("task")
        if task is None:
            # Fallback: call the original handler; if it crashes on 'task',
            # print a minimal message instead so real errors are visible.
            try:
                _original_exc_handler(self, context)
            except (KeyError, AttributeError):
                msg = context.get("message", "<no message>")
                print(
                    f"AsyncSlicer: unhandled exception in event loop: {msg}"
                    + (f" ({exc!r})" if exc else "")
                )
        else:
            _original_exc_handler(self, context)

    _QAsyncioEventLoop.default_exception_handler = _safe_default_exception_handler  # type: ignore[attr-defined]

except ImportError:
    pass  # Not running under PySide6 / QtAsyncio — patches are not needed.
# ---------------------------------------------------------------------------

# Callable that takes a ChunkRequest and returns a coroutine yielding an ndarray.
FetchFn = Callable[[ChunkRequest], Coroutine[Any, Any, np.ndarray]]

# Callable fired once per batch with (request, data) pairs.
BatchCallback = Callable[[list[tuple[ChunkRequest, np.ndarray]]], None]


class AsyncSlicer:
    """Generic cancellable async batch-fetch service.

    Owns one ``asyncio.Task`` per active ``slice_request_id``.  The data
    source is **not** held as state; a ``fetch_fn`` coroutine is supplied
    on each ``submit()`` call so the same slicer instance can serve any
    storage backend or chunk type (volumes, images, point clouds).

    All reads within a batch are issued concurrently via ``asyncio.gather``.
    After each batch the callback fires and ``asyncio.sleep(0)`` yields to
    Qt so the renderer can flush pending GPU uploads and redraw.

    Parameters
    ----------
    batch_size :
        Number of chunks to read concurrently before yielding to Qt.
        Higher -> fewer render interruptions and faster total load.
        Lower -> more frequent visual feedback.  Default 8 is a good
        balance for local NVMe / SSD storage.
    render_every :
        Yield to Qt (triggering a render) only every this many batches.
        The final batch always triggers a render regardless.
        Default 1 renders after every batch.  Set higher (e.g. 4) to
        reduce intermediate render overhead at the cost of less frequent
        visual feedback during loading.
    """

    def __init__(self, batch_size: int = 8, render_every: int = 1) -> None:
        self._batch_size = batch_size
        self._render_every = max(1, render_every)
        # Maps slice_request_id -> running asyncio.Task.
        self._tasks: dict[UUID, asyncio.Task] = {}

    # ── Public API ──────────────────────────────────────────────────────

    def submit(
        self,
        requests: list[ChunkRequest],
        fetch_fn: FetchFn,
        callback: BatchCallback,
        consumer_id: str | None = None,
    ) -> UUID | None:
        """Submit a batch of chunk requests for async loading.

        All requests must share the same ``slice_request_id``.  If a task
        for that ID is already running it is cancelled first (fire-and-forget;
        ``CancelledError`` is re-raised inside the task so asyncio marks it
        properly).

        Parameters
        ----------
        requests :
            Ordered list of ``ChunkRequest`` objects, all sharing one
            ``slice_request_id``.  Empty list is a no-op (returns ``None``).
        fetch_fn :
            Async callable ``(ChunkRequest) -> np.ndarray``.  Typically a
            bound method such as ``data_store.get_data``.  Captured in the
            task closure so different submit calls can use different stores.
        callback :
            Called once per batch with a list of ``(request, data)`` pairs.
            Runs on the Qt main thread (QtAsyncio single-thread model).
            Responsible for committing data to the GPU cache and
            rebuilding the LUT.
        consumer_id :
            Optional label for logging and future priority routing.

        Returns
        -------
        slice_request_id :
            The ``UUID`` shared by all submitted requests, or ``None`` if
            ``requests`` was empty.  Store this to pass to ``cancel()``
            before the next ``submit()`` call.
        """
        if not requests:
            return None

        slice_id = requests[0].slice_request_id

        # Safety net: cancel any lingering task for this exact slice ID.
        # In normal operation the app calls cancel() explicitly before
        # submitting a fresh batch, but this guards against accidental
        # same-ID resubmission.
        if slice_id in self._tasks:
            self._tasks[slice_id].cancel()

        task = asyncio.ensure_future(self._run(requests, fetch_fn, callback, slice_id))
        self._tasks[slice_id] = task

        _SLICER_LOGGER.info(
            "task_submitted  requests=%d  slice_id=%s  consumer=%r",
            len(requests),
            slice_id,
            consumer_id,
        )

        return slice_id

    def cancel(self, slice_request_id: UUID | None) -> bool:
        """Cancel in-flight work for ``slice_request_id``.

        No-op if the ID is unknown or the task has already finished.

        Parameters
        ----------
        slice_request_id :
            The ID returned by a previous ``submit()`` call.

        Returns
        -------
        cancelled :
            ``True`` if a running task was found and cancelled;
            ``False`` otherwise.  The app uses this to decide whether to
            log a cancellation event in the debug summary.
        """
        if slice_request_id is None:
            return False
        task = self._tasks.pop(slice_request_id, None)
        if task is not None and not task.done():
            task.cancel()
            return True
        return False

    # ── Internal coroutine ──────────────────────────────────────────────

    async def _run(
        self,
        requests: list[ChunkRequest],
        fetch_fn: FetchFn,
        callback: BatchCallback,
        slice_id: UUID,
    ) -> None:
        """Drive the batched read loop.

        Splits ``requests`` into batches of ``self._batch_size``, calls
        ``fetch_fn`` for all chunks in each batch concurrently via
        ``asyncio.gather``, fires the callback, then yields to Qt.

        ``CancelledError`` is always re-raised so asyncio marks the task
        as cancelled.  Partially-completed batches at the time of
        cancellation are silently discarded -- the callback will not fire
        for the in-progress batch.

        The ``finally`` block removes ``slice_id`` from ``self._tasks``
        unconditionally (normal completion, exception, or cancellation).
        We use ``finally`` rather than ``add_done_callback`` because
        ``QAsyncioTask`` does not implement ``_make_cancelled_error``, a
        CPython-internal method that ``asyncio`` invokes when processing
        done-callbacks on cancelled futures.
        """
        batches = [
            requests[i : i + self._batch_size]
            for i in range(0, len(requests), self._batch_size)
        ]
        n_batches = len(batches)

        _SLICER_LOGGER.info(
            "task_start  slice_id=%s  total_requests=%d  n_batches=%d",
            slice_id,
            len(requests),
            n_batches,
        )

        _cancelled = False
        batch_idx = 0
        batch_times_ms: list[float] = []
        t_fetch_start = time.perf_counter()

        try:
            for batch_idx, batch in enumerate(batches):
                # asyncio.gather issues all reads in the batch concurrently,
                # allowing tensorstore to pipeline chunk fetches.
                #
                # The PySide6 compatibility patch applied at module load time
                # (see above) adds the missing _make_cancelled_error() method
                # to QAsyncioTask, which is required by gather's internal
                # _GatheringFuture._done_callback when child tasks are cancelled.
                t_batch = time.perf_counter()
                results: list[np.ndarray] = await asyncio.gather(
                    *[fetch_fn(req) for req in batch]
                )
                batch_ms = (time.perf_counter() - t_batch) * 1000
                batch_times_ms.append(batch_ms)

                # Per-batch fetch timing at DEBUG.
                _PERF_LOGGER.debug(
                    "fetch_batch  %d/%d  bricks=%d  elapsed=%.1fms",
                    batch_idx + 1,
                    n_batches,
                    len(batch),
                    batch_ms,
                )

                # Condensed batch summary at INFO.
                if _SLICER_LOGGER.isEnabledFor(logging.INFO):
                    scale_counts: dict[int, int] = {}
                    for req in batch:
                        scale_counts[req.scale_index] = (
                            scale_counts.get(req.scale_index, 0) + 1
                        )
                    _SLICER_LOGGER.info(
                        "batch_done  %d/%d  bricks=%d  scales=%s",
                        batch_idx + 1,
                        n_batches,
                        len(batch),
                        scale_counts,
                    )

                # Per-brick detail at DEBUG (guarded).
                if _SLICER_LOGGER.isEnabledFor(logging.DEBUG):
                    for req, data in zip(batch, results):
                        _SLICER_LOGGER.debug(
                            "  brick_received  id=%s  scale=%d  shape=%s",
                            req.chunk_request_id,
                            req.scale_index,
                            data.shape,
                        )

                callback(list(zip(batch, results)))
                # Yield to Qt every render_every batches so the renderer can
                # flush pending GPU uploads and redraw.  Always yield on the
                # final batch so the completed state is always visible.
                is_last_batch = batch_idx == n_batches - 1
                if is_last_batch or (batch_idx + 1) % self._render_every == 0:
                    await asyncio.sleep(0)

        except asyncio.CancelledError:
            _cancelled = True
            _SLICER_LOGGER.info(
                "task_cancelled  slice_id=%s  batches_done=%d/%d",
                slice_id,
                batch_idx,
                n_batches,
            )
            raise  # mandatory -- marks the task as cancelled

        finally:
            # Always remove from the live-task dict, whether we completed
            # normally, were cancelled, or raised an unexpected exception.
            self._tasks.pop(slice_id, None)

            # Emit fetch timing summary (both normal completion and
            # cancellation — partial stats are still useful).
            if batch_times_ms:
                total_ms = (time.perf_counter() - t_fetch_start) * 1000
                _log_fetch_summary(
                    slice_id,
                    len(requests),
                    batch_times_ms,
                    total_ms,
                    cancelled=_cancelled,
                )

            if not _cancelled:
                _SLICER_LOGGER.info(
                    "task_complete  slice_id=%s  total_requests=%d",
                    slice_id,
                    len(requests),
                )


def _log_fetch_summary(
    slice_id: UUID,
    n_requests: int,
    batch_times_ms: list[float],
    total_ms: float,
    *,
    cancelled: bool,
) -> None:
    """Emit an INFO-level fetch timing summary on ``_PERF_LOGGER``."""
    n = len(batch_times_ms)
    mean = sum(batch_times_ms) / n
    min_t = min(batch_times_ms)
    max_t = max(batch_times_ms)
    if n > 1:
        variance = sum((t - mean) ** 2 for t in batch_times_ms) / (n - 1)
        std = math.sqrt(variance)
    else:
        std = 0.0

    status = "cancelled" if cancelled else "complete"
    _PERF_LOGGER.info(
        "fetch_summary  status=%s  slice_id=%s  bricks=%d  batches=%d  "
        "total=%.0fms  per_batch=%.1f\u00b1%.1fms  min=%.1fms  max=%.1fms",
        status,
        slice_id,
        n_requests,
        n,
        total_ms,
        mean,
        std,
        min_t,
        max_t,
    )
