"""Shared immutable state snapshots — used by events and render layer."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Mapping

#: The empty thickness mapping, shared because it is immutable.
NO_THICKNESS: Mapping[int, float] = MappingProxyType({})


class AxisAlignedSelectionState(NamedTuple):
    """Immutable snapshot of a scene's displayed axes and slab thickness.

    **There are no slice positions here.**  A ``DimsChangedEvent`` carries
    them in its own ``slice_indices`` field, and the editable positions are
    on the scene's ``dims.selection`` (an ``AxisAlignedSelection``).

    Attributes
    ----------
    displayed_axes : tuple[int, ...]
        The world axes the view shows, in display order.
    thickness : Mapping[int, float]
        Per world axis, the half-thickness of the slab around the slice, in
        world units.  An axis absent from it has thickness 0, a plane.
    """

    displayed_axes: tuple[int, ...]
    thickness: Mapping[int, float] = NO_THICKNESS


class DimsState(NamedTuple):
    """A scene's dims: axis names and which axes are shown where.

    Attributes
    ----------
    axis_labels : tuple[str, ...]
        The name of each world axis, in world order.
    selection : AxisAlignedSelectionState
        The displayed axes and the per-axis slab thickness.  The slice
        positions are not here: see ``DimsChangedEvent.slice_indices``.
    """

    axis_labels: tuple[str, ...]
    selection: AxisAlignedSelectionState


class CameraState(NamedTuple):
    """A snapshot of a canvas's active camera.

    Positions and directions are in the renderer's ``(x, y, z)`` frame: x is
    the last displayed world axis, so a scene showing ``("z", "y", "x")``
    has renderer x = world x and renderer z = world z.  Pass it back to
    ``CellierController.set_camera_state`` to restore a view.

    Attributes
    ----------
    camera_type : str
        ``"perspective"`` for a 3D view, ``"orthographic"`` for a 2D one.
    position : tuple[float, float, float]
        Camera position.
    rotation : tuple[float, float, float, float]
        Orientation as a quaternion ``(x, y, z, w)``.
    up : tuple[float, float, float]
        The camera's up vector; always ``(0, 1, 0)`` for a 2D camera.
    fov : float
        Vertical field of view in degrees; ``0`` for an orthographic camera.
    zoom : float
        Zoom factor.
    extent : tuple[float, float]
        The visible ``(width, height)`` in world units for an orthographic
        camera; ``(0, 0)`` for a perspective one.
    depth_range : tuple[float, float] or None
        The ``(near, far)`` clipping distances, or ``None`` when the range
        is automatic (worked out from what the camera is fitted to).
    """

    camera_type: Literal["perspective", "orthographic"]
    position: tuple[float, float, float]
    rotation: tuple[float, float, float, float]
    up: tuple[float, float, float]
    fov: float
    zoom: float
    extent: tuple[float, float]
    # ``None`` is an automatic range: pygfx works it out from what the
    # camera is fitted to.  A 2D camera has one until a range is set.
    depth_range: tuple[float, float] | None
