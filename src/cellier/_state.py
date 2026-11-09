"""Shared immutable state snapshots — used by events and render layer."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Mapping

#: The empty thickness mapping, shared because it is immutable.
NO_THICKNESS: Mapping[int, float] = MappingProxyType({})


class AxisAlignedSelectionState(NamedTuple):
    """Immutable snapshot of an axis-aligned selection.

    ``thickness`` is a per-axis world-unit **half**-thickness; an axis absent
    from it has thickness 0, a plane.

    **There are no slice positions here.**  Every consumer takes the
    ``RegionSelection`` the controller emits instead of a mapping of world
    axis to world position: it says the same thing in a form that survives a
    slab, a viewport crop or an oblique plane, and it is pulled back through
    each visual's own transform rather than compared against data coordinates
    as though it were already in them.  The editable positions are still on
    ``cellier.scene.dims.AxisAlignedSelection``, which is what the sliders
    write to and what the region is built from.
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
        The displayed axes and every axis's position and thickness.
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
