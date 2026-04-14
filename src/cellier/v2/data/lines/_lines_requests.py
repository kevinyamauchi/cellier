# src/cellier/v2/data/lines/_lines_requests.py
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from uuid import UUID

    import numpy as np


class LinesSliceRequest(NamedTuple):
    """Request for one slab-filtered slice of line-segment data.

    The first three fields satisfy the AsyncSlicer logging contract:
    ``slice_request_id`` is the task key; ``chunk_request_id`` and
    ``scale_index`` appear in INFO/DEBUG log lines.

    Parameters
    ----------
    slice_request_id : UUID
        Shared ID for all requests in one planning event.  Used by
        AsyncSlicer as the dict key for the running task — REQUIRED.
    chunk_request_id : UUID
        Per-request ID.  For lines (never tiled) this equals
        ``slice_request_id``.
    scale_index : int
        Always 0 — no LOD levels.  Present for slicer logging compat.
    displayed_axes : tuple[int, ...]
        Axis indices rendered in the canvas (2 for 2D, 3 for 3D).
    slice_indices : dict[int, int]
        Collapsed axis → world-space integer slice position.
        Empty when all axes are displayed (full 3D view).
    thickness : float
        Half-thickness of the slab in data-space voxel units.  A segment
        is included when **both** of its endpoints satisfy
        ``slice_index - thickness <= coord <= slice_index + thickness``
        on every non-displayed axis.  Default 0.5.
    """

    slice_request_id: UUID
    chunk_request_id: UUID
    scale_index: int
    displayed_axes: tuple[int, ...]
    slice_indices: dict[int, int]
    thickness: float = 0.5


@dataclass(frozen=True)
class LinesData:
    """Slab-filtered line-segment data returned by LinesMemoryStore.get_data().

    Parameters
    ----------
    request_id : UUID
        Echo of LinesSliceRequest.slice_request_id.
    positions : np.ndarray
        (n_vertices, n_displayed_dims) float32.  n_vertices is always even;
        pair ``(2n, 2n+1)`` defines segment ``n``.  Projected onto displayed
        axes; padded to 3D in the render layer.
    colors : np.ndarray | None
        (n_vertices, 4) float32 RGBA, index-matched to positions.
        None when the store carries no per-vertex colors.
    color_mode : str
        ``"uniform"`` or ``"vertex"``.  Ignored when colors is None.
    is_empty : bool
        True when the slab filter produced zero surviving segments and a
        placeholder geometry was returned.
    """

    request_id: UUID
    positions: np.ndarray
    colors: np.ndarray | None
    color_mode: str = "uniform"
    is_empty: bool = False

    @property
    def shape(self) -> str:
        """Summary string consumed by AsyncSlicer DEBUG logging."""
        return f"positions={self.positions.shape}"
