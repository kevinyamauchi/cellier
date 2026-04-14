# src/cellier/v2/data/mesh/_mesh_requests.py
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from uuid import UUID

    import numpy as np


class MeshSliceRequest(NamedTuple):
    """Request for one slab-filtered slice of mesh data.

    The first three fields satisfy the AsyncSlicer logging contract:
    ``slice_request_id`` is the task key; ``chunk_request_id`` and
    ``scale_index`` appear in INFO/DEBUG log lines.

    Parameters
    ----------
    slice_request_id : UUID
        Shared ID for all requests in one planning event.  Used by
        AsyncSlicer as the dict key for the running task — REQUIRED.
    chunk_request_id : UUID
        Per-request ID.  For mesh (never tiled) this equals
        ``slice_request_id``.
    scale_index : int
        Always 0 — no LOD levels.  Present for slicer logging compat.
    displayed_axes : tuple[int, ...]
        Axis indices rendered in the canvas.
    slice_indices : dict[int, int]
        Collapsed axis → world-space integer slice position.
        Empty when all axes are displayed (full 3D view).
    thickness : float
        Half-thickness of the slab in data-space units.  Faces whose
        at least one vertex satisfies
        ``slice_index - thickness <= coord <= slice_index + thickness``
        on every sliced axis are included.  Default 0.5.
    """

    slice_request_id: UUID
    chunk_request_id: UUID
    scale_index: int
    displayed_axes: tuple[int, ...]
    slice_indices: dict[int, int]
    thickness: float = 0.5


@dataclass(frozen=True)
class MeshData:
    """Filtered, reindexed mesh data returned by MeshMemoryStore.get_data().

    Parameters
    ----------
    request_id : UUID
        Echo of MeshSliceRequest.slice_request_id.
    positions : np.ndarray
        (n_vertices, n_displayed_dims) float32.  Projected onto
        displayed axes; padded to 3D in the render layer.
    indices : np.ndarray
        (n_faces, 3) int32.  Reindexed to reference only the vertices
        in ``positions``.
    normals : np.ndarray
        (n_vertices, n_displayed_dims) float32.  Projected normals.
    colors : np.ndarray | None
        Per-vertex (n_vertices, 4) or per-face (n_faces, 4) float32
        RGBA.  None when the store carries no color data.
    color_mode : str
        ``"vertex"`` or ``"face"``.  Ignored when colors is None.
    is_empty : bool
        True when the slab contained no surviving faces and a
        placeholder geometry was returned.
    """

    request_id: UUID
    positions: np.ndarray
    indices: np.ndarray
    normals: np.ndarray
    colors: np.ndarray | None
    color_mode: str = "vertex"
    is_empty: bool = False

    @property
    def shape(self) -> str:
        """Summary string consumed by AsyncSlicer DEBUG logging."""
        return f"positions={self.positions.shape} " f"indices={self.indices.shape}"
