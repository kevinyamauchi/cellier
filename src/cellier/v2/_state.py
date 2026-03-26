"""Shared immutable state snapshots — used by events and render layer."""

from __future__ import annotations

from typing import Literal, NamedTuple


class AxisAlignedSelectionState(NamedTuple):
    """Immutable snapshot of an axis-aligned selection."""

    displayed_axes: tuple[int, ...]
    slice_indices: dict[int, int]

    def to_index_selection(self, ndim: int) -> tuple[int | slice, ...]:
        """Return a per-axis numpy indexer in axis order.

        Displayed axes → ``slice(None)``, sliced axes → their int value.
        """
        result: list[int | slice] = []
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
