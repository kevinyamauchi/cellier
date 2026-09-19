# src/cellier/v2/data/mesh/_mesh_requests.py
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from uuid import UUID

    import numpy as np

    from cellier.transform import ConvexRegion


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
    retained_axes : tuple[int, ...]
        The **data** axes this visual's geometry keeps, ascending.

        Not the same list as ``displayed_axes``, which indexes the **world**:
        a ``zyx`` store in a ``czyx`` world retains ``(0, 1, 2)`` while the
        world displays ``(1, 2, 3)``, and a transform that permutes its axes
        retains a different set again.  Indexing the position array with
        world axes raises on the first and silently uploads the wrong columns
        on the second.
    region : ConvexRegion
        The selected region, already pulled back into **data** coordinates
        (design 3.12).  It is the whole filter -- one ``contains`` call
        instead of a per-axis mask loop.

        Until v1 was retired this was optional, and a request without one
        fell back to comparing a **world** position from ``slice_indices``
        against **data** coordinates -- the latent bug D4 exists to fix: on a
        2 um z spacing it could show a vertex 24 um off the slice plane and
        hide the ones that are on it.  There is no second path now (R8.3).
    """

    slice_request_id: UUID
    chunk_request_id: UUID
    scale_index: int
    displayed_axes: tuple[int, ...]
    retained_axes: tuple[int, ...]
    region: ConvexRegion


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
    original_face_indices : np.ndarray | None
        (n_faces,) int array mapping each row of ``indices`` back to its face
        index in the store's full face array.  This is the slab filter's
        surviving-face list; in a full 3-D view it is the identity
        ``arange(n_faces)``.  Used by the render layer to translate a pick's
        rendered face index into the original face index.  None on the
        empty-placeholder path.
    """

    request_id: UUID
    positions: np.ndarray
    indices: np.ndarray
    normals: np.ndarray
    colors: np.ndarray | None
    color_mode: str = "vertex"
    is_empty: bool = False
    original_face_indices: np.ndarray | None = None

    @property
    def shape(self) -> str:
        """Summary string consumed by AsyncSlicer DEBUG logging."""
        return f"positions={self.positions.shape} indices={self.indices.shape}"
