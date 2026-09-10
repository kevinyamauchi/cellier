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

    ``slice_indices`` values are **world-space positions**, not voxel
    indices, and are floats since D3.  ``thickness`` is a per-axis world-unit
    **half**-thickness; an axis absent from it uses
    ``cellier.scene.dims.DEFAULT_HALF_THICKNESS``.
    """

    displayed_axes: tuple[int, ...]
    slice_indices: dict[int, float]
    stacked_axes: tuple[int, ...] = ()
    thickness: Mapping[int, float] = NO_THICKNESS

    def to_index_selection(self, ndim: int) -> tuple[float | slice, ...]:
        """Return a per-axis numpy indexer in axis order.

        Displayed axes -> ``slice(None)``, sliced axes -> their value.
        """
        result: list[float | slice] = []
        for axis in range(ndim):
            if axis in self.slice_indices:
                result.append(self.slice_indices[axis])
            else:
                result.append(slice(None))
        return tuple(result)


class PlaneSelectionState(NamedTuple):
    """Stub — not yet implemented."""

    pass


SelectionState = AxisAlignedSelectionState | PlaneSelectionState


class DimsState(NamedTuple):
    """Current dimension display state for a scene."""

    axis_labels: tuple[str, ...]
    selection: SelectionState


class CameraState(NamedTuple):
    """Immutable snapshot of the active camera's logical state."""

    camera_type: Literal["perspective", "orthographic"]
    position: tuple[float, float, float]
    rotation: tuple[float, float, float, float]
    up: tuple[float, float, float]
    fov: float
    zoom: float
    extent: tuple[float, float]
    depth_range: tuple[float, float]
