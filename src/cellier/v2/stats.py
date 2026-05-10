"""Performance statistics dataclasses for cellier v2.

All types are plain dataclasses (no psygnal).  The Qt widget polls or
subscribes to events to refresh the display rather than reacting to
field mutations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uuid import UUID


@dataclass
class RenderStats:
    """EMA-smoothed render performance statistics for one canvas.

    Parameters
    ----------
    alpha :
        EMA blend weight used once warm-up is complete.  During warm-up
        the weight is ``1 / frame_count`` so the first frame sets the
        value directly and subsequent frames converge quickly.  Must be
        in ``(0, 1]``.

    Attributes
    ----------
    fps : float
        Smoothed frames-per-second estimate.
    draw_time_ms : float
        Smoothed CPU draw-submission time in milliseconds.
    last_updated : float
        ``time.perf_counter()`` timestamp of the most recent update.
        Zero until the first frame is recorded.
    """

    alpha: float = 0.1
    fps: float = 0.0
    draw_time_ms: float = 0.0
    last_updated: float = 0.0
    _frame_count: int = field(default=0, repr=False)

    def update(self, frame_ms: float, now: float) -> None:
        """Record one frame and update EMA estimates.

        Parameters
        ----------
        frame_ms :
            Wall time of the frame's CPU submission in milliseconds.
        now :
            ``perf_counter()``-compatible timestamp for ``last_updated``.
            Pass the end-of-frame timestamp already held by the caller to
            avoid a redundant syscall.
        """
        self._frame_count += 1
        # Warm-up: use 1/n until it drops below alpha, then clamp.
        # Mirrors the TemporalAccumulationPass warm-up strategy.
        a = max(self.alpha, 1.0 / self._frame_count)
        self.draw_time_ms = a * frame_ms + (1.0 - a) * self.draw_time_ms
        if frame_ms > 0.0:
            instant_fps = 1000.0 / frame_ms
            self.fps = a * instant_fps + (1.0 - a) * self.fps
        self.last_updated = now


@dataclass
class FetchStats:
    """Timing snapshot for the most recent completed fetch task on one visual.

    All times are in milliseconds.  Updated once per completed (or
    cancelled) ``AsyncSlicer`` task; persists until the next fetch so
    the widget always shows the last known values.

    Attributes
    ----------
    n_chunks : int
        Number of chunk requests in the task.
    total_ms : float
        Total elapsed time from first fetch to last callback.
    ms_per_chunk : float
        ``total_ms / n_chunks`` — mean wall time per chunk.
    cancelled : bool
        ``True`` if the task was cancelled before completion.
    last_updated : float
        ``time.perf_counter()`` timestamp of the most recent update.
        Zero until the first fetch completes.
    """

    n_chunks: int = 0
    total_ms: float = 0.0
    ms_per_chunk: float = 0.0
    cancelled: bool = False
    last_updated: float = 0.0


@dataclass
class PerformanceStats:
    """Container for all performance statistics.

    Owned by ``RenderManager`` and exposed via
    ``CellierController.stats``.

    Attributes
    ----------
    render : dict[UUID, RenderStats]
        Per-canvas render statistics, keyed by canvas UUID.
        An entry is created when ``RenderManager.add_canvas`` is called.
    fetch : dict[UUID, FetchStats]
        Per-visual fetch statistics, keyed by visual model UUID.
        An entry is created on the first completed fetch for that visual.
    """

    render: dict[UUID, RenderStats] = field(default_factory=dict)
    fetch: dict[UUID, FetchStats] = field(default_factory=dict)
