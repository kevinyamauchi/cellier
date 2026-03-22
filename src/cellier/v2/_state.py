"""Shared immutable state snapshots — used by events and render layer."""

from __future__ import annotations

from typing import Literal, NamedTuple


class DimsState(NamedTuple):
    """Current dimension display state for a scene."""

    displayed_axes: tuple[int, ...]
    slice_indices: tuple[int, ...]


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
