# src/cellier/v2/data/points/_points_requests.py
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from uuid import UUID

    import numpy as np


class PointsSliceRequest(NamedTuple):
    """Request for one proximity-filtered slice of points data.

    The first three fields satisfy the AsyncSlicer logging contract:
    ``slice_request_id`` is the task key; ``chunk_request_id`` and
    ``scale_index`` appear in INFO/DEBUG log lines.

    Parameters
    ----------
    slice_request_id : UUID
        Shared ID for all requests in one planning event.  Used by
        AsyncSlicer as the dict key for the running task — REQUIRED.
    chunk_request_id : UUID
        Per-request ID.  For points (never tiled) this equals
        ``slice_request_id``.
    scale_index : int
        Always 0 — no LOD levels.  Present for slicer logging compat.
    displayed_axes : tuple[int, ...]
        Axis indices rendered in the canvas (2 for 2D, 3 for 3D).
    slice_indices : dict[int, int]
        Collapsed axis → world-space integer slice position.
        Empty when all axes are displayed (full 3D view).
    thickness : float
        Half-thickness of the proximity slab in data-space voxel units.
        A point on a non-displayed axis ``a`` is included when
        ``slice_indices[a] - thickness <= coord[a] <= slice_indices[a] + thickness``.
        Default 0.5 (one voxel either side of the slice plane).
    """

    slice_request_id: UUID
    chunk_request_id: UUID
    scale_index: int
    displayed_axes: tuple[int, ...]
    slice_indices: dict[int, int]
    thickness: float = 0.5


@dataclass(frozen=True)
class PointsData:
    """Proximity-filtered points data returned by PointsMemoryStore.get_data().

    Parameters
    ----------
    request_id : UUID
        Echo of PointsSliceRequest.slice_request_id.
    positions : np.ndarray
        (n_points, n_displayed_dims) float32.  Projected onto displayed
        axes; padded to 3D in the render layer.
    colors : np.ndarray | None
        (n_points, 4) float32 RGBA, index-matched to positions.
        None when the store carries no per-point colors.
    sizes : np.ndarray | None
        (n_points,) float32 per-point sizes.
        None when the store carries no per-point sizes.
    color_mode : str
        ``"uniform"`` or ``"vertex"``.  Ignored when colors is None.
    size_mode : str
        ``"uniform"`` or ``"vertex"``.  Ignored when sizes is None.
    is_empty : bool
        True when the proximity filter produced zero surviving points
        and a placeholder geometry was returned.
    """

    request_id: UUID
    positions: np.ndarray
    colors: np.ndarray | None
    sizes: np.ndarray | None
    color_mode: str = "uniform"
    size_mode: str = "uniform"
    is_empty: bool = False

    @property
    def shape(self) -> str:
        """Summary string consumed by AsyncSlicer DEBUG logging."""
        return f"positions={self.positions.shape}"
