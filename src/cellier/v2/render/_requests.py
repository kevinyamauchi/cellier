"""Immutable data-transfer objects for the render layer request pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from uuid import UUID

    import numpy as np


class DimsState(NamedTuple):
    """Current dimension display state.

    Parameters
    ----------
    displayed_axes : tuple[int, ...]
        Indices of axes currently displayed, e.g. ``(0, 1, 2)`` for a 3D
        view or ``(0, 1)`` for a 2D view.
    slice_indices : tuple[int, ...]
        Current slice index for each non-displayed axis, in the same order
        as the non-displayed axes.
    """

    displayed_axes: tuple[int, ...]
    slice_indices: tuple[int, ...]


class ReslicingRequest(NamedTuple):
    """Frozen snapshot driving a complete reslicing cycle.

    Constructed by ``CanvasView.capture_reslicing_request()`` and consumed
    by ``SliceCoordinator``.  All array fields must be ``.copy()``d by the
    caller — the ``NamedTuple`` does not enforce this.

    Parameters
    ----------
    camera_pos : ndarray, shape (3,)
        World-space camera position, copy.
    frustum_corners : ndarray, shape (2, 4, 3)
        World-space frustum corners, copy.
    fov_y_rad : float
        Vertical field of view in radians.
    screen_height_px : float
        Logical screen height in pixels, baked in at snapshot time.
    dims_state : DimsState
        Current dimension display state.
    request_id : UUID
        Unique identifier per trigger; used for cancellation.
    scene_id : UUID
        Which scene this camera belongs to.
    target_visual_ids : frozenset[UUID] or None
        ``None`` means reslice all visuals in the scene (camera-moved case).
        A non-None set means reslice only those specific visuals
        (data-updated case).
    """

    camera_pos: np.ndarray
    frustum_corners: np.ndarray
    fov_y_rad: float
    screen_height_px: float
    dims_state: DimsState
    request_id: UUID
    scene_id: UUID
    target_visual_ids: frozenset[UUID] | None
